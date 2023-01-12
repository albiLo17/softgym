import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import Picker
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy
from softgym.utils.misc import quatFromAxisAngle
# Add needed environmental paths
import os
os.environ['PYFLEXROOT'] = os.environ['PWD'] + "/PyFlex"
os.environ['LD_LIBRARY_PATH'] = os.environ['PYFLEXROOT'] + "/external/SDL2-2.0.4/lib/x64"

class SoftNewEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=75, render_mode='particle', picker_radius=0.02, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb', 'key_point']
        assert action_mode in ['picker', 'sawyer', 'franka']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.num_picker = num_picker

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, picker_threshold=0.005,
            particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.action_space = self.action_tool.action_space
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = 10 * 3
            else:
                max_particles = 41
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode in ['picker']:
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        self.horizon = horizon
        # print("init of rope new env done!")

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'paramNum': 4,          # What is this?
            'ClusterSpacing': 1.0,
            'ClusterRadius': 0.0,
            'ClusterStiffness': 0.5,
            'dynamicFriction': 1.5,
            'particleFriction': 1.0,
            'ClusterPlasticThreshold': 0.0,
            'ClusterPlasticCreep': 0.0,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self.key_point_indices, :3]
            pos = keypoint_pos.flatten()
            # more_info = np.array([self.rope_length, self._get_endpoint_distance()])
            # pos = np.hstack([more_info, pos])
            # print("in _get_obs, pos is: ", pos)

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def _get_key_point_idx(self, num=None):
        indices = [0]
        interval = (num - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)
        indices.append(num - 1)

        return indices

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2

        camera_params = config['camera_params'][config['camera_name']]
        params = np.array(
            [config['paramNum'], config['ClusterSpacing'], config['ClusterRadius'], config['ClusterStiffness'], config['dynamicFriction'], config['particleFriction'],
                config['ClusterPlasticThreshold'],
                config['ClusterPlasticCreep'], *camera_params['angle'][:], camera_params['width'], camera_params['height'], render_mode]
            )

        env_idx = 6

        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (params, robot_params)
            pyflex.set_scene(env_idx, params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, params, 0)

        num_particles = pyflex.get_n_particles()
        # print("with {} segments, the number of particles are {}".format(config['segment'], num_particles))
        # exit()
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def _get_info(self):
        return {}

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def _reset(self,):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1,p2), :3] # Was changed from p1, p4.
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]]) # take p1

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.5, 0., -0.5], [0.5, 0.5, 0.5])
            self.action_tool.set_picker_pos(picker_pos=key_point_pos[0] + np.array([0., picker_radius, 0.]))


            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)


        config = self.get_current_config()

        pyflex.step()

        info = self._get_info()
        return self._get_obs()



if __name__ == '__main__':
    env = SoftNewEnv(observation_mode='key_point',
                  action_mode='picker',
                  num_picker=2,
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=1,
                  use_cached_states=False,
                  save_cached_states=False,
                  deterministic=False)
    env.reset(config=env.get_default_config())
    for i in range(1000):
        print(i)
        print("right before pyflex step")
        pyflex.step()
        print("right after pyflex step")
        print("right before pyflex render")
        pyflex.render()
        print("right after pyflex render")
