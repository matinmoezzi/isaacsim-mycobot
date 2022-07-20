
# envs/vec_env_rlgames.py
from omni.isaac.gym.vec_env import VecEnvBase
import torch

import numpy as np
import hydra
from omegaconf import DictConfig
# these two functions are designed to read yaml files (get config parameters)
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

class VecEnvRLGames(VecEnvBase): # connecting RL policies with task implementations
    def _process_data(self): # not inherited
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True) -> None: # inherited
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def step(self, actions): # not inherited but exist in  VecEnvBase
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


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict) # dict

    headless = cfg.headless # False
    render = not headless

    env = VecEnvRLGames(headless=headless)

    from omniisaacgymenvs.mycobot.mycobot_random_policy_affiliated import myCobotTask
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(cfg_dict) # return config as <class 'omniisaacgymenvs.utils.config_utils.sim_config.SimConfig'>
    cfg = sim_config.config

    task = myCobotTask(name="myCobot", sim_config=sim_config, env=env)
    physics_params = sim_config.get_physics_params()
    # print(physics_params)
    env.set_task(task=task, sim_params=physics_params, backend="torch", init_sim=True) # init_sim=init_sim

    # while env._simulation_app.is_running():
    #     if env._world.is_playing():
    #         if env._world.current_time_step_index == 0:
    #             env._world.reset(soft=True)
    #         actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #         env._task.pre_physics_step(actions)
    #         env._world.step(render=render)
    #         env.sim_frame_count += 1
    #         env._task.post_physics_step()
    #     else:
    #         env._world.step(render=render)

    # env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()

# https://forums.developer.nvidia.com/t/cannot-import-name-physxscheam-from-pxr/190357
# physics_params = {'gravity': [0.0, 0.0, -9.81], 'dt': 0.0083, 'substeps': 1, 'use_gpu_pipeline': True, 'add_ground_plane': True, 'add_distant_light': True, 'use_flatcache': True, 'enable_scene_query_support': False, 'default_physics_material': {'static_friction': 1.0, 'dynamic_friction': 1.0, 'restitution': 0.0}, 'use_gpu': True, 'worker_thread_count': 4, 'solver_type': 1, 'bounce_threshold_velocity': 0.2, 'friction_offset_threshold': 0.04, 'friction_correlation_distance': 0.025, 'enable_sleeping': True, 'enable_stabilization': True, 'gpu_max_rigid_contact_count': 524288, 'gpu_max_rigid_patch_count': 81920, 'gpu_found_lost_pairs_capacity': 1024, 'gpu_found_lost_aggregate_pairs_capacity': 262144, 'gpu_total_aggregate_pairs_capacity': 1024, 'gpu_max_soft_body_contacts': 1048576, 'gpu_max_particle_contacts': 1048576, 'gpu_heap_capacity': 67108864, 'gpu_temp_buffer_capacity': 16777216, 'gpu_max_num_partitions': 8, 'solver_position_iteration_count': 4, 'solver_velocity_iteration_count': 0, 'sleep_threshold': 0.0, 'stabilization_threshold': 0.0, 'enable_gyroscopic_forces': False, 'density': 1000.0, 'max_depenetration_velocity': 100.0, 'contact_offset': 0.02, 'rest_offset': 0.001, 'sim_device': 'cuda:0'}