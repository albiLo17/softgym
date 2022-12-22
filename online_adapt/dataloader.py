import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import yaml
import glob
import h5py
import utils
import open3d as o3d

from softgym.utils.visualization import save_numpy_as_gif, save_numpy_to_gif_matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pcd(pcd, elev=30, azim=-180):
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(projection='3d', elev=elev, azim=azim)
    # rotate by 90, invert
    img = ax.scatter(pcd[:,2], pcd[:,0], pcd[:,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0, 0.9)

    plt.show()
class PointcloudDataset(Dataset):
    def __init__(self, dataset_folders, num_past=3, num_pred=1, train=True):
        super().__init__()

        self.data_folders = dataset_folders

        self.train = train

        self.num_past = num_past      # number of future predcitions
        self.num_pred = num_pred      # number of past observations

        self.voxel_size = 0.03
        self.max_points = 0

        self.look_up = []
        self.data = self.load_data()
        self.num_datapoints = len(self.look_up)

        # Pad zeros to pcds that have different dimensions than the biggest one
        self.pad_pcds()

        print()

    def load_data(self):
        self.params = []
        self.pcds = []
        self.actions = []
        env = 0
        for folder in tqdm(self.data_folders):
            try:
                param, pcd, action = self.load_pcd(folder)
                self.params.append(param)
                self.pcds.append(pcd)
                self.actions.append(action)

                tot_len = len(pcd) - self.num_past - self.num_pred
                num_segments = tot_len
                for seg in range(num_segments):
                    # Plus one as we want also the observation (but not the action) of the next step
                    self.look_up.append({'env': env,
                                         'past': slice(self.num_past * seg, self.num_past * (seg + 1) + 1),
                                         'pred': slice(self.num_past * (seg + 1), self.num_past * (seg + 1) + self.num_pred + 1)})
                env += 1
            except RuntimeError as e:
                print(f'Folder unstable: {folder}')

    def downsample_pcd(self, pcd):
        # voxel_filter = o3d.geometry.VoxelGrid()
        # voxel_filter.voxel_size = voxel_size
        # downsampled_pcd = voxel_filter.filter(pcd)

        vector3d_vector = o3d.utility.Vector3dVector(pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = vector3d_vector
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        # Get all the voxels in the voxel grid
        voxels_all = voxel_grid.get_voxels()
        # get all the centers and colors from the voxels in the voxel grid
        all_centers = [voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels_all]
        downsampled_pcd = np.asarray(all_centers)

        return downsampled_pcd


    def load_pcd(self, path):
        samples = glob.glob(path + '/*')
        samples.sort()

        params = None
        pcds = []
        actions = []
        for s in samples:
            # TODO: set correct names
            f = h5py.File(s, 'r')
            if params is None:
                # stiff = np.asarray(f.get('ClothStiff'))
                # size = np.asarray(f.get('ClothSize'))     # TODO: check size
                # mass = np.asarray(f.get('mass'))
                # friction = np.asarray(f.get('dynamic_friction'))

                mass = np.asarray(f.get('ClothStiff'))
                stiff = np.asarray(f.get('dynamic_friction'))
                friction =np.asarray(f.get('ClothSize'))
                params = np.append(stiff, friction)
                params = np.append(params, mass)

            pcd_pos = np.asarray(f.get('pcd_pos'))
            # pcds.append(pcd_pos)
            downsampled_pcd_pos = self.downsample_pcd(pcd_pos)
            pcds.append(downsampled_pcd_pos)
            if downsampled_pcd_pos.shape[0] > self.max_points:
                self.max_points = downsampled_pcd_pos.shape[0]

            # action = np.asarray(f.get('action'))[:3]
            action = np.asarray(f.get('vel'))[:3]
            actions.append(action)
            f.close()

        return params, pcds, actions

    def pad_pcds(self):
        for e, env in enumerate(self.pcds):
            for i, pcd in enumerate(env):
                temp = np.zeros((self.max_points, 3))
                temp[:pcd.shape[0], :] = pcd
                self.pcds[e][i] = temp

    # def visualize_points(self, pos, edge_index=None, index=None):
    #     fig = plt.figure(figsize=(4, 4))
    #     if edge_index is not None:
    #         for (src, dst) in edge_index.t().tolist():
    #             src = pos[src].tolist()
    #             dst = pos[dst].tolist()
    #             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    #     if index is None:
    #         plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    #     else:
    #         mask = torch.zeros(pos.size(0), dtype=torch.bool)
    #         mask[index] = True
    #         plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
    #         plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    #     plt.axis('off')
    #     plt.show()

    def _get_datapoint(self, idx):

        # TODO: augment datapoints

        env = self.look_up[idx]['env']
        range_past = self.look_up[idx]['past']
        range_pred = self.look_up[idx]['pred']

        # (p_t-past, ..., p_t-1, p_t, a_t-past, ..., a_t-1)
        past_pcd = torch.stack([torch.from_numpy(a) for a in self.pcds[env][range_past]])
        past_a = torch.stack([torch.from_numpy(a) for a in self.actions[env][range_past][1:]])    # Start from 1 beacuse the points are coupled as (a_t, pcd_t+1)

        # (p_t, ..., p_t+future, a_t, ..., a_t+future-1)
        future_pcd = torch.stack([torch.from_numpy(a) for a in self.pcds[env][range_pred]])
        future_a = torch.stack([torch.from_numpy(a) for a in self.actions[env][range_pred][1:]])

        params = torch.from_numpy(self.params[env]).float()

        return past_pcd, past_a, future_pcd, future_a, params


    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
         return self.num_datapoints



if __name__ == '__main__':
    path = '../examples/data/env*'
    paths = glob.glob(path)
    paths.sort()

    dataset = PointcloudDataset(paths[:10])
    dataset._get_datapoint(3)
    print()