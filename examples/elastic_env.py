import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
import random
from matplotlib import pyplot as plt

import h5py

# Add needed environmental paths
import os
os.environ['PYFLEXROOT'] = os.environ['PWD'] + "/PyFlex"
os.environ['LD_LIBRARY_PATH'] = os.environ['PYFLEXROOT'] + "/external/SDL2-2.0.4/lib/x64"


import matplotlib.pyplot as plt


def generate_trajectory(waypoints, deviation):
    # Generate a trajectory by interpolating the waypoints with random deviation[x, y, z]
    trajectory = [waypoints[0]]
    actions = []
    for i in range(len(waypoints) - 1):
        # x1 = waypoints[i][0] + random.uniform(-deviation[0], deviation[0])
        # y1 = waypoints[i][1] + random.uniform(-deviation[1], deviation[1])
        # z1 = waypoints[i][2] + random.uniform(-deviation[2], deviation[2])
        x2 = waypoints[i + 1][0] + random.uniform(-deviation[0], deviation[0])
        y2 = waypoints[i + 1][1] + random.uniform(-deviation[1], deviation[1])
        z2 = waypoints[i + 1][2] + random.uniform(-deviation[2], deviation[2])
        # num_points = int(n / (len(waypoints)-1))
        # for j in range(num_points):
        #     t = 1
        #     if num_points > 1:
        #         t = j / (num_points - 1)
        #     x = x1 * (1 - t) + x2 * t
        #     y = y1 * (1 - t) + y2 * t
        #     z = z1 * (1 - t) + z2 * t

        actions.append([x2 - trajectory[-1][0], y2 - trajectory[-1][1], z2 - trajectory[-1][2], 1.])
        trajectory.append([x2, y2, z2])    # The last 1 is because we always want to keep the particle grasped

    return trajectory, actions

def store_data_by_name(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()

def load_data(path):
    f = h5py.File(path, 'r')
    obs = np.asarray(f.get('obs'))
    f.close()
    return obs

def make_dir(path):
    tot_path = ''
    for folder in path.split('/'):
        if not folder == '.' and not folder == '':
            tot_path = tot_path + folder + '/'
            if not os.path.exists(tot_path):
                os.mkdir(tot_path)
                # print(tot_path)
        else:
            if folder == '.':
                tot_path = tot_path + folder + '/'
def plot_pcd(pcd):
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(projection='3d')
    # rotate by 90, invert
    img = ax.scatter(pcd[:,2], pcd[:,0], pcd[:,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0, 0.9)

    plt.show()


def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrag')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    parser.add_argument('--save_data', type=bool, default=False, help='save trajectory in a folder')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['observation_mode '] = 'point_cloud'      # cam_rgb, point_cloud, key_point
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')


    # #
    # part:
    stiff_configs = [[0.8, 1, 0.9, 0.75, 1.3, 0.2],  [0.8, 1, 0.9, 0.75, 1.5, 20.]]


    env = normalize(SOFTGYM_ENVS[args.env_name](cloth_stiff=stiff_configs, **env_kwargs))

    for env_idx in range(args.num_variations):
        print(f'Env: {env_idx}')
        if args.save_data:
            data_save_path = f'./data/env_{env_idx}'
            make_dir(data_save_path)

        # env.reset(config_id=env_idx)
        env.reset(config_id=env_idx)

        config = env.get_current_config()
        frames = [env.get_image(args.img_size, args.img_size)]
        pcds = [env._get_obs().reshape(-1,3)[:env.particle_num, :]]

        # Define actions along trajectories
        vel = random.uniform(0.7, 1.3)     #[1, 2, ]
        x_w = 0.1 + random.uniform(-0.02, -0.02)
        y_w = 0.05 + random.uniform(-0.01, 0.01)
        waypoints = [[x_w*i*vel, y_w*i*vel, 0.*i*vel] for i in range((int(env.horizon/vel)) + 1)]      # +1 because we need to interpolate so we want n+1 points
        # deviation = [0., 0., 0.]
        deviation = [0.05, 0.01, 0.001]
        trajectory, actions = generate_trajectory(waypoints=waypoints, deviation=deviation)
        for i in range(len(actions)):
            # action = np.asarray([0.1, 0.05, 0., 1.]*2)

            # a = list(env.action_space.sample()[:4]*0.5)
            # a[-1] = 1.
            # action = np.asarray(a * 2)

            action = np.asarray(actions[i] * 2)

            # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
            # intermediate frames. Only use this option for visualization as it increases computation.
            obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)

            frames.extend(info['flex_env_recorded_frames'])

            if args.save_data:
                save_name = "data_{:06}".format(i)
                pcd_pos = obs.reshape(-1,3)[:env.particle_num, :]
                pcd_vel = pcd_pos - pcds[-1]
                store_data_by_name(['pcd_pos', 'pcd_vel', 'mass', 'ClothStiff', 'dynamic_friction', 'ClothSize', 'action', 'vel'],
                                   [pcd_pos, pcd_vel, config['mass'], config['ClothStiff'], config['dynamic_friction'], config['ClothSize'], action, vel],
                                   osp.join(data_save_path, save_name))
            # plot_pcd(env._get_obs().reshape(-1, 3))
            if args.test_depth:
                show_depth()

            pcds.append(obs.reshape(-1,3)[:env.particle_num, :])

        if args.save_data:
            save_name = osp.join(args.save_video_dir, args.env_name + f'_env_{i}')

        if args.save_video_dir is not None:
            # params = env._wrapped_env.get_current_config()['ClothStiff']
            params = stiff_configs[env_idx]
            save_name = osp.join(args.save_video_dir, args.env_name + f'V3_det_stretch_{params[0]}_bend_{params[1]}_shear_{params[2]}_envf_{params[4]}_partf_{params[5]}_mass_{params[3]}.gif')
            save_numpy_as_gif(np.array(frames), save_name)
            print('Video generated and save to {}'.format(save_name))




    print()

if __name__ == '__main__':
    main()

    # stiff_configs = [[0.1, 0.1, 0.1], [1.5, 0.1, 0.1], [3., 0.1, 0.1],      # Stretch, Bend, Shear
    #                  [0.1, 1., 0.1], [1.5, 1., 0.1], [3., 1., 0.1],
    #                  [0.1, 2., 0.1], [1.5, 2., 0.1], [3., 2., 0.1],
    #
    #                  [0.1, 0.1, 0.5], [1.5, 0.1, 0.5], [3., 0.1, 0.5],  # Stretch, Bend, Shear
    #                  [0.1, 1., 0.5], [1.5, 1., 0.5], [3., 1., 0.5],
    #                  [0.1, 2., 0.5], [1.5, 2., 0.5], [3., 2., 0.5],
    #
    #                  [0.1, 0.1, 1.], [1.5, 0.1, 1.], [3., 0.1, 1.],  # Stretch, Bend, Shear
    #                  [0.1, 1., 1.], [1.5, 1., .1], [3., 1., 1.],
    #                  [0.1, 2., 1.], [1.5, 2., 1.], [3., 2., 1.],
    #
    #                  ]
    #
    # stiff_configs = [[0.1, 0.1, 1.]]
    # # stiff_configs = [[1.5, 0.1, 1.]]
    # # stiff_configs = [[3., 0.1, 1.]]
    # # # #
    # # stiff_configs = [[0.1, 1., 1.]]
    # # stiff_configs = [[1.5, 1., 1.]]
    # # stiff_configs = [[3., 1, 1.]]
    # # # #
    # # stiff_configs = [[0.1, 2., 1.]]
    # # stiff_configs = [[1.5, 2., 1.]]
    # # stiff_configs = [[3., 2., 1.]]
    #
    # stiff_configs = [[0.1, 0.1, 0.1, 0.75, 1.], [4., 1., 1., 0.75, 2.],  [4., 3., 1., 3., 1.]]
    # stiff_configs = [[2., 2., 1., 0.75, 1.], [2., 2., 1., 0.01, 1.], [2., 2., 1., 10., 1.]]
    # stiff_configs = [[0.1, 1., 1., 0.75, 1., 0.2], [0.1, 1., 1., 1.5, 1., 0.2], [0.1, 1., 1.6, 2., 1., 0.2]]
    #
    # # stretch:
    # stiff_configs = [[0.1, 1, 0.9, 0.75, 1.5, 0.2],  [3., 1, 0.9, 0.75, 1.5, 0.2]]
    #
    # # bend:
    # stiff_configs = [[0.8, 0.1, 0.9, 0.75, 1.5, 0.2],  [0.8, 2, 0.9, 0.75, 1.5, 0.2]]
    # #
    # # shear:
    # stiff_configs = [[0.8, 1, 0.1, 0.75, 1.5, 0.2],  [0.8, 1, 0.9, 0.75, 1.5, 0.2]]
    # # #
    # # mass:
    # stiff_configs = [[0.8, 1, 0.9, 0.1, 1.5, 0.2],  [0.8, 1, 0.9, 3., 1.5, 0.2]]
    # #
    # # dyn:
    # stiff_configs = [[0.8, 1, 0.9, 0.75, 1.4, 0.2],  [0.8, 1, 0.9, 0.75, 1.5, 0.2]]
