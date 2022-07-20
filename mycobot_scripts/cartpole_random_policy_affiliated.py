# tasks/rl_task.py
from abc import abstractmethod # sim files
import numpy as np
import torch
from gym import spaces
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner
from omniisaacgymenvs.tasks.utils.usd_utils import create_distant_light
import omni.kit

class RLTask(BaseTask):

    """ This class provides a PyTorch RL-specific interface for setting up RL tasks. 
        It includes utilities for setting up RL task related parameters,
        cloning environments, and data collection for RL algorithms.
    """

    def __init__(self, name, env, offset=None) -> None:

        """ Initializes RL parameters, cloner object, and buffers.

        Args:
            name (str): name of the task.
            env (VecEnvBase): an instance of the environment wrapper class to register task.
            offset (Optional[np.ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        super().__init__(name=name, offset=offset)

        self.test = self._cfg["test"]
        self._device = self._cfg["sim_device"]
        print("Task Device:", self._device)

        self.clip_obs = self._cfg["task"]["env"].get("clipObservations", np.Inf)
        self.clip_actions = self._cfg["task"]["env"].get("clipActions", np.Inf)
        self.rl_device = self._cfg.get("rl_device", "cuda:0")

        self.control_frequency_inv = self._cfg["task"]["env"].get("controlFrequencyInv", 1)

        print("RL device: ", self.rl_device)

        self._env = env

        if not hasattr(self, "_num_agents"):
            self._num_agents = 1  # used for multi-agent environments
        if not hasattr(self, "_num_states"):
            self._num_states = 0

        # initialize data spaces (defaults to gym.Box)
        if not hasattr(self, "action_space"):
            self.action_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        if not hasattr(self, "observation_space"):
            self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        if not hasattr(self, "state_space"):
            self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self.cleanup()

    def cleanup(self) -> None:
        """ Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self._num_envs, self.num_observations), device=self._device, dtype=torch.float)
        self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def set_up_scene(self, scene) -> None:
        """ Clones environments based on value provided in task config and applies collision filters to mask 
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
        """

        super().set_up_scene(scene)

        collision_filter_global_paths = list()
        if self._sim_config.task_config["sim"].get("add_ground_plane", True):
            self._ground_plane_path = "/World/defaultGroundPlane"
            collision_filter_global_paths.append(self._ground_plane_path)
            scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths)
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path, "/World/collisions", prim_paths, collision_filter_global_paths)
        self.set_initial_camera_params(camera_position=[10, 10, 3], camera_target=[0, 0, 0])
        if self._sim_config.task_config["sim"].get("add_distant_light", True):
            create_distant_light()
    
    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    @property
    def default_base_env_path(self):
        """ Retrieves default path to the parent of all env prims.

        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        """ Retrieves default path to the first env prim (index 0).

        Returns:
            default_zero_env_path(str): Defaults to "/World/envs/env_0".
        """
        return f"{self.default_base_env_path}/env_0"

    @property
    def num_envs(self):
        """ Retrieves number of environments for task.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    @property
    def num_actions(self):
        """ Retrieves dimension of actions.

        Returns:
            num_actions(int): Dimension of actions.
        """
        return self._num_actions

    @property
    def num_observations(self):
        """ Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_observations

    @property
    def num_states(self):
        """ Retrieves dimesion of states.

        Returns:
            num_states(int): Dimension of states.
        """
        return self._num_states

    @property
    def num_agents(self):
        """ Retrieves number of agents for multi-agent environments.

        Returns:
            num_agents(int): Dimension of states.
        """
        return self._num_agents

    def get_states(self):
        """ API for retrieving states buffer, used for asymmetric AC training.

        Returns:
            states_buf(torch.Tensor): States buffer.
        """
        return self.states_buf

    def get_extras(self):
        """ API for retrieving extras data for RL.

        Returns:
            extras(dict): Dictionary containing extras data.
        """
        return self.extras

    def reset(self):
        """ Flags all environments for reset.
        """
        self.reset_buf = torch.ones_like(self.reset_buf)

    def pre_physics_step(self, actions):
        """ Optionally implemented by individual task classes to process actions.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        pass

    def post_physics_step(self):
        """ Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


# robots/articulation/cartpole.py
from typing import Optional # sim files
# import numpy as np
# import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb

class Cartpole(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Cartpole",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )


# envs/cartpole.py
# from omniisaacgymenvs.tasks.base.rl_task import RLTask
# from omniisaacgymenvs.robots.articulations.cartpole import Cartpole

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

# import numpy as np
# import torch
import math

class CartpoleTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        super().set_up_scene(scene)
        self._cartpoles = ArticulationView(prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view")
        scene.add(self._cartpoles)
        return

    def get_cartpole(self):
        cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs_buf[:, 0] = cart_pos
        self.obs_buf[:, 1] = cart_vel
        self.obs_buf[:, 2] = pole_pos
        self.obs_buf[:, 3] = pole_vel

        observations = {
            self._cartpoles.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        cart_pos = self.obs_buf[:, 0]
        pole_pos = self.obs_buf[:, 2]

        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
