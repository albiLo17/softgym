import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt

# Add needed environmental paths
import os
os.environ['PYFLEXROOT'] = os.environ['PWD'] + "/PyFlex"
os.environ['LD_LIBRARY_PATH'] = os.environ['PYFLEXROOT'] + "/external/SDL2-2.0.4/lib/x64"


import matplotlib.pyplot as plt

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
    parser.add_argument('--num_variations', type=int, default=3, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')

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

    stiff_configs = [[0.1, 0.1, 0.1], [1.5, 0.1, 0.1], [3., 0.1, 0.1],      # Stretch, Bend, Shear
                     [0.1, 1., 0.1], [1.5, 1., 0.1], [3., 1., 0.1],
                     [0.1, 2., 0.1], [1.5, 2., 0.1], [3., 2., 0.1],

                     [0.1, 0.1, 0.5], [1.5, 0.1, 0.5], [3., 0.1, 0.5],  # Stretch, Bend, Shear
                     [0.1, 1., 0.5], [1.5, 1., 0.5], [3., 1., 0.5],
                     [0.1, 2., 0.5], [1.5, 2., 0.5], [3., 2., 0.5],

                     [0.1, 0.1, 1.], [1.5, 0.1, 1.], [3., 0.1, 1.],  # Stretch, Bend, Shear
                     [0.1, 1., 1.], [1.5, 1., .1], [3., 1., 1.],
                     [0.1, 2., 1.], [1.5, 2., 1.], [3., 2., 1.],

                     ]

    stiff_configs = [[0.1, 0.1, 1.]]
    # stiff_configs = [[1.5, 0.1, 1.]]
    # stiff_configs = [[3., 0.1, 1.]]
    # # #
    # stiff_configs = [[0.1, 1., 1.]]
    # stiff_configs = [[1.5, 1., 1.]]
    # stiff_configs = [[3., 1, 1.]]
    # # #
    # stiff_configs = [[0.1, 2., 1.]]
    # stiff_configs = [[1.5, 2., 1.]]
    # stiff_configs = [[3., 2., 1.]]

    stiff_configs = [[0.1, 0.1, 0.1, 0.75, 1.], [4., 1., 1., 0.75, 2.],  [4., 3., 1., 3., 1.]]
    stiff_configs = [[2., 2., 1., 0.75, 1.], [2., 2., 1., 0.75, 10.], [2., 2., 1., 3., 1.]]


    env = normalize(SOFTGYM_ENVS[args.env_name](cloth_stiff=stiff_configs, **env_kwargs))

    for env_idx in range(args.num_variations):
        env.reset(config_id=env_idx)


        frames = [env.get_image(args.img_size, args.img_size)]
        for i in range(env.horizon):
            action = np.asarray([0.1, 0.05, 0., 1.]*2)
            # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
            # intermediate frames. Only use this option for visualization as it increases computation.
            _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
            frames.extend(info['flex_env_recorded_frames'])
            # plot_pcd(env._get_obs().reshape(-1, 3))
            if args.test_depth:
                show_depth()

        # if args.save_video_dir is not None:
        #     params = env._wrapped_env.get_current_config()['ClothStiff']
        #     save_name = osp.join(args.save_video_dir, args.env_name + f'_{params[0]}_{params[1]}_{params[2]}.gif')
        #     save_numpy_as_gif(np.array(frames), save_name)
        #     print('Video generated and save to {}'.format(save_name))


    print()

if __name__ == '__main__':
    main()
