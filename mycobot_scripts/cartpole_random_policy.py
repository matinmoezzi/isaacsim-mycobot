
# utils/task_util.py
def initialize_task(config, env, init_sim=True):
    # from omniisaacgymenvs.tasks.allegro_hand import AllegroHandTask
    # from omniisaacgymenvs.tasks.ant import AntLocomotionTask
    # from omniisaacgymenvs.tasks.cartpole import CartpoleTask
    # from omniisaacgymenvs.tasks.humanoid import HumanoidLocomotionTask
    # from omniisaacgymenvs.tasks.shadow_hand import ShadowHandTask


    # Mappings from strings to environments
    # task_map = {
    #     "AllegroHand": AllegroHandTask,
    #     "Ant": AntLocomotionTask,
    #     "Cartpole": CartpoleTask,
    #     "Humanoid": HumanoidLocomotionTask,
    #     "ShadowHand": ShadowHandTask,
    # }
    from omniisaacgymenvs.mycobot.cartpole_random_policy_affiliated import CartpoleTask

    task_map = {"Cartpole": CartpoleTask}

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task


# envs/vec_env_rlgames.py
from omni.isaac.gym.vec_env import VecEnvBase
import torch

class VecEnvRLGames(VecEnvBase): # connecting RL policies with task implementations
    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def step(self, actions):
        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()
        self._task.pre_physics_step(actions)

        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()
        self._states = self._task.get_states()
        self._process_data()
        
        obs_dict = {"obs": self._obs, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict


# main function containing the whole sequence
import numpy as np
# import torch
import hydra
from omegaconf import DictConfig

# these two functions are designed to read yaml files (get config parameters)
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

# from omniisaacgymenvs.utils.task_util import initialize_task
# from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless # False
    render = not headless

    
    env = VecEnvRLGames(headless=headless)
    task = initialize_task(cfg_dict, env)

    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)
            actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
            env._task.pre_physics_step(actions)
            env._world.step(render=render)
            env.sim_frame_count += 1
            env._task.post_physics_step()
        else:
            env._world.step(render=render)

    env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()

# https://forums.developer.nvidia.com/t/cannot-import-name-physxscheam-from-pxr/190357

