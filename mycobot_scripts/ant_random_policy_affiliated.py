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

#tasks/shared/locomotion.py
from abc import abstractmethod

#from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math


class LocomotionTask(RLTask):
    def __init__(
        self,
        name,
        env,
        offset=None
    ) -> None:

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]

        RLTask.__init__(self, name, env)
        return

    @abstractmethod
    def set_up_scene(self, scene) -> None:
        pass

    @abstractmethod
    def get_robot(self):
        pass

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # force sensors attached to the feet
        sensor_force_torques = self._robots._physics_view.get_force_sensor_forces() # (num_envs, num_sensors, 6)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = get_observations(
            torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.targets, self.potentials, self.dt,
            self.inv_start_rot, self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale
        )
        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        forces = self.actions * self.joint_gears * self.power_scale

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        # applies joint torques
        self._robots.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._robots.num_dof), device=self._device)
        dof_pos[:] = tensor_clamp(
            self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper
        )
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = calculate_metrics(
            self.obs_buf, self.actions, self.up_weight, self.heading_weight, self.potentials, self.prev_potentials,
            self.actions_cost_scale, self.energy_cost_scale, self.termination_height,
            self.death_cost, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio
        )

    def is_done(self) -> None:
        self.reset_buf[:] = is_done(
            self.obs_buf, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    targets,
    potentials,
    dt,
    inv_start_rot,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    num_envs,
    contact_force_scale,
    actions,
    angular_velocity_scale
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            normalize_angle(angle_to_target).unsqueeze(-1),
            up_proj.unsqueeze(-1),
            heading_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
            actions,
        ),
        dim=-1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec


@torch.jit.script
def is_done(
    obs_buf,
    termination_height,
    reset_buf,
    progress_buf,
    max_episode_length
):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tensor
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reset


@torch.jit.script
def calculate_metrics(
    obs_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(
        obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8
    )

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )
    return total_reward




#robots/articulations/ant.py
from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb


class Ant(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Ant",
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
            self._usd_path = assets_root_path + "/Isaac/Robots/Ant/ant_instanceable.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )



#tasks/ant.py
#from omniisaacgymenvs.robots.articulations.ant import Ant
#from omniisaacgymenvs.test_files.shared.locomotion import LocomotionTask
#from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

from pxr import PhysxSchema

import numpy as np
import torch
import math

#inherents from LocomotionTask
class AntLocomotionTask(LocomotionTask):
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
        self._num_observations = 60
        self._num_actions = 8
        self._ant_positions = torch.tensor([0, 0, 0.5])

        LocomotionTask.__init__(self, name=name, env=env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_ant()
        RLTask.set_up_scene(self, scene)
        self._ants = ArticulationView(prim_paths_expr="/World/envs/.*/Ant/torso", name="ant_view")
        scene.add(self._ants)
        return

    def get_ant(self):
        ant = Ant(prim_path=self.default_zero_env_path + "/Ant", name="Ant", translation=self._ant_positions)
        self._sim_config.apply_articulation_settings("Ant", get_prim_at_path(ant.prim_path), self._sim_config.parse_actor_config("Ant"))

    def get_robot(self):
        return self._ants

    def post_reset(self):
        self.joint_gears = torch.tensor([15, 15, 15, 15, 15, 15, 15, 15], dtype=torch.float32, device=self._device)
        dof_limits = self._ants.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self._device)

        LocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self._ants.num_dof)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, num_dof):
    # type: (Tensor, int) -> Tensor
    return torch.sum(obs_buf[:, 12:12+num_dof] > 0.99, dim=-1)