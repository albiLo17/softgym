import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
from softgym.utils.misc import quatFromAxisAngle


class ClothFoldElasEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_fold_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            # 'ClothSize': [64, 32],
            'ClothSize': [34, 34],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0,
            'dynamic_friction': 1.5,
            'particle_friction': 1.0,
            'mass': 0.5
        }
        return config

    def generate_env_variation(self,
                               num_variations=2,
                               vary_cloth_size=False,
                               vary_cloth_params=False,
                               ):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            if vary_cloth_params:
                stiffness = [np.random.uniform(0.5, 3), np.random.uniform(0.1, 2.), np.random.uniform(0.5, 0.9)]
                cloth_mass = np.random.uniform(0.2, 2.0)
                dynamic_friction = np.random.uniform(1.4, 1.7)
                particle_friction = 1.
                config.update({
                    'ClothStiff': stiffness,
                    'mass': cloth_mass,
                    # 'mesh_verts': mesh_verts.reshape(-1),
                    # 'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
                    # 'mesh_bend_edges': mesh_bend_edges.reshape(-1),
                    # 'mesh_shear_edges': mesh_shear_edges.reshape(-1),
                    # 'mesh_faces': mesh_faces.reshape(-1),
                    'dynamic_friction': dynamic_friction,
                    'particle_friction': particle_friction
                })

            self.set_scene(config, box=False)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))

            # compute box params
            # center = np.array([0.2, 0.25, 0.1])
            # quat = quatFromAxisAngle([0, 0, -1.], 0.)
            # halfEdge = np.array([0.05, 0.05, 0.05])
            # pyflex.add_box(halfEdge, center, quat)

            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()
            angle = (np.random.random() - 0.5) * np.pi / 2
            self.rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            # print('config {}: {}'.format(i, config['camera_params']))
            print('config {}: ClothStiff: {}, mass: {}, dyn friction: {}, part friction: {}'.format(i, config['ClothStiff'], config['mass'], config['dynamic_friction'], config['particle_friction'] ))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def _reset(self, box=False):
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

        if box:
            # compute box params
            center = np.array([0.0, -0.04, -0.0])
            quat = quatFromAxisAngle([0, 0, -1.], 0.)
            halfEdge = np.array([0.8, 0.05, 0.8])
            pyflex.add_box(halfEdge, center, quat)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            print(self.action_tool.next_action)
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]

        p1, p2, p3, p4 = self._get_key_point_idx()
        key_point_pos = pos[(p1, p2, p3, p4), :3]

        # pos_group_a = pos[self.fold_group_a]
        # pos_group_b = pos[self.fold_group_b]
        # pos_group_b_init = self.init_pos[self.fold_group_b]
        # curr_dist = np.mean(np.linalg.norm(key_point_pos[:2] - key_point_pos[2:], axis=1))
        curr_dist = np.mean(np.linalg.norm(key_point_pos[0] - key_point_pos[1])) + np.mean(np.linalg.norm(key_point_pos[2] - key_point_pos[3]))
        reward = -curr_dist
        return reward

    def eval_done(self, threshold=0.001):
        distance = - self.compute_reward()
        if distance <= threshold:
            return True
        return False


    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def _set_to_folded(self):
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        fold_group_a = particle_grid_idx[:, :x_split].flatten()
        fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        curr_pos[fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.

        pyflex.set_positions(curr_pos)
        for i in range(10):
            pyflex.step()
        return self._get_info()['performance']
