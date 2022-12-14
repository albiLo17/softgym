import numpy as np
import argparse
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_box_env import ClothBoxEnv
from softgym.utils.misc import quatFromAxisAngle
import copy
from copy import deepcopy


class ClothDragEnv(ClothBoxEnv):
    def __init__(self, cached_states_path='cloth_drop_init_states.pkl',
                 cloth_stiff=[0.9, 1.0, 0.9],
                 **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        self.stiff_config = cloth_stiff
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        assert self.action_tool.num_picker == 2  # Two drop points for this task
        self.prev_dist = None  # Should not be used until initialized
        self.config = None

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            # 'ClothSize': [64, 32],
            'ClothSize': [64, 64],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0,
            'dynamic_friction': 0.75,
            'particle_friction': 1.0,
        }
        return config

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < 0.5:
            curr_pos[:, 1] += 0.5 - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)
        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _sample_cloth_size(self):
        return np.random.randint(60, 100), np.random.randint(60, 100)

    def generate_env_variation(self, num_variations=1,
                               vary_cloth_size=False,
                               vary_cloth_params=False,
                               ):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
                config.update({
                    'ClothStiff': self.stiff_config[i],
                })
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
                config.update({
                    'ClothStiff': self.stiff_config[i][:3],
                    'dynamic_friction': self.stiff_config[i][3],
                    'particle_friction': self.stiff_config[i][4]
                })


            # Stretch, Bend and Shear
            # TODO: add mass (flingbot)
            if vary_cloth_params:
                stiffness = np.random.uniform(0.85, 0.95, 3)
                cloth_mass = np.random.uniform(0.2, 2.0)
                dynamic_friction = np.random.uniform(0.1, 0.95)
                particle_friction = np.random.uniform(0.85, 0.95)
                config.update({
                    'ClothStiff': stiffness,
                    # 'mass': cloth_mass,
                    # 'mesh_verts': mesh_verts.reshape(-1),
                    # 'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
                    # 'mesh_bend_edges': mesh_bend_edges.reshape(-1),
                    # 'mesh_shear_edges': mesh_shear_edges.reshape(-1),
                    # 'mesh_faces': mesh_faces.reshape(-1),
                    'dynamic_friction': dynamic_friction,
                    'particle_friction': particle_friction
                })

            #self.config = config
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            config['target_pos'] = self._get_flat_pos()
            # low_val = np.random.random()        # Before. TODO: remember to set it back
            low_val = 0.5
            self._set_to_vertical(x_low=low_val * 0.2 - 0.1, height_low=low_val * 0.1 + 0.1)

            # Get height of the cloth without the gravity. With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()

            curr_pos = pyflex.get_positions().reshape(-1, 4)
            curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))
        return generated_configs, generated_states

    # def set_scene(self, config, states=None):
    #     super().set_scene(config)
    #
    #     # compute box params
    #     center = np.array([0., 0., 0.])
    #     quat = quatFromAxisAngle([0, 0, -1.], 0.)
    #     halfEdge = np.array([0.1, 0.1, 0.1])
    #     pyflex.add_box(halfEdge, center, quat)
    #
    #     self.box_states = np.array([0, 0, 0.])
    #     pyflex.set_shape_states(self.box_states)

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.5, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))
            # self.action_tool.visualize_picker_boundary()
        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        pyflex.step()
        return

    def _get_current_dist(self, pos):
        target_pos = self.get_current_config()['target_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - target_pos, axis=1))
        return curr_dist

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        r = - curr_dist
        return r

    def _get_info(self):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        performance = -curr_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        return {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init)}



