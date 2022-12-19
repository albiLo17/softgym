import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import yaml
import glob
import h5py
import utils

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
    def __init__(self, dataset_folders,):
        super().__init__()
        print()

    def load_pcd(self, path):
        samples = glob.glob(path + '/*')
        samples.sort()
        pcds = []
        for s in samples:
            f = h5py.File(s, 'r')
            obs = np.asarray(f.get('obs')).reshape(-1, 3)
            pcds.append(obs)
            f.close()

        return pcds

    def visualize_points(self, pos, edge_index=None, index=None):
        fig = plt.figure(figsize=(4, 4))
        if edge_index is not None:
            for (src, dst) in edge_index.t().tolist():
                src = pos[src].tolist()
                dst = pos[dst].tolist()
                plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
        if index is None:
            plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
        else:
            mask = torch.zeros(pos.size(0), dtype=torch.bool)
            mask[index] = True
            plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
            plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
        plt.axis('off')
        plt.show()



class DeformableDataset(Dataset):
    def __init__(self, dataset_folders, args, transform=None, structured=False, shuffle=True, train=False):
        super().__init__()

        self.task = args.task
        self.args = args
        self.train = train

        self.M = args.M
        self.K = args.K

        # Equal to M makes it easier to return the sequence of the task for the prediction
        self.data_frequency = args.traj_frequency
        self.delta_t = self.M       # parameter that defines the interval between two sequences of time t_1, t_2
        self.transform = transform

        # Dataset contraints. Hopefully this won't be necessary anymore
        stream = open(args.dataset_config_file, 'r')
        self.constraints = yaml.load(stream, Loader=yaml.FullLoader)

        # TODO: normalization?
        # self.mean = {'pulling': {'action': [], 's1': [], 's2': [], 'k': [], 'b': []},
        #              'task': {'action': [], 's1': [], 's2': [], 'k': [], 'b': []},
        #              'tot': {'action': [], 's1': [], 's2': [], 'k': [], 'b': []}}

        # Memorize weighted adjacencies of batch if oracle
        self.adjs = []
        self.k = []
        self.b = []
        self.scale_params = args.scale_params
        self.current_adj = []

        # Load tasks
        self.data = self.load_data(dataset_folders, args)
        self.n_tasks = len(self.data)

        self.look_up = []
        self.look_up_tasks = []
        self.process_data()
        self.num_datapoints = len(self.look_up)


        # print()

    def load_data(self, dataset_folders, args):
        data = None
        task = args.task

        # Define init and start point to load
        s = self.constraints[task][2][0]
        e = self.constraints[task][2][1]

        self.init_graphs = []
        for folder in dataset_folders:
            k, b = folder.split('/')[-1].split('_')
            k = float(k.replace('elas', ''))
            b = float(b.replace('bend', ''))
            self.k.append(k)
            self.b.append(b)

            tmp_data = process_datapoint(
                data=np.load(folder + '/dataset.npy', allow_pickle=True),
                k=k,
                b=b,
                ratio=args.graph_scale,
                start_idx=s,
                end_idx=e,
                frequency=self.data_frequency
            )

            # data.append(tmp_data)
            tmp_data = np.expand_dims(tmp_data, 0)
            data = np.concatenate([data, tmp_data[0:1]], 0) if data is not None else tmp_data[0:1]
            # self.num_datapoints += tmp_data.shape[1]
            # self.start_task_idx.append(self.num_datapoints)

        return data

    def process_data(self):
        for task in tqdm(range(len(self.data))):

            task_data = self.data[task]

            for i, datapoint in enumerate(task_data):
                if len(datapoint) == 4:
                    state1, edge1, action, force_1 = datapoint


                edge1 = edge1[:, :3]

                # Store weighted adjacency
                if len(self.adjs) < task + 1:
                    scale_k = 0.1  # The normal range is form 15 to 80, so scale by 0.1
                    self.adjs.append(utils.part_anis_to_adj(edge1, state1.shape[0], k=self.k[task], b=self.b[task], scale=self.scale_params, order=1))


                state_normalization = 1
                action_normalization = 1

                state1 = torch.from_numpy(state1).float() / state_normalization
                edge1 = torch.from_numpy(edge1).long().T
                action = torch.from_numpy(action).float() / action_normalization

                force_1 = torch.from_numpy(force_1).float()

                if len(edge1) == 0:
                    edge1 = torch.empty(2, 0, dtype=torch.long)

                # Use this in case in the edges is provided the value of k
                data1 = Data(x=state1, edge_index=edge1[:-1])

                if self.transform:
                    data1 = self.transform(data1)


                processed_datapoint = [data1.x, data1.adj, action, force_1]

                self.data[task][i] = processed_datapoint

            # define intervals for the task
            init_length_lookup = len(self.look_up)
            tot_len = task_data.shape[0] - self.M - self.K
            num_segments = int(tot_len / self.delta_t)
            for seg in range(num_segments):
                # Plus one as we want also the observation (but not the action) of the next step
                self.look_up.append({'task': task,
                                     'M': torch.arange(self.M*seg,self.M*(seg+1)+1),
                                     'K': torch.arange(self.M*(seg+1),self.M*(seg+1) + self.K+1)})

            final_length_lookup = len(self.look_up)
            # Create look up also for the entire trajectory
            self.look_up_tasks.append(torch.arange(init_length_lookup, final_length_lookup))

    def get_task_traj(self, task):
        traj = [self._get_datapoint(segment) for segment in self.look_up_tasks[task]]
        return traj

    def _get_datapoint(self, idx):

        # TODO: augment datapoints

        task = self.look_up[idx]['task']
        range_past = self.look_up[idx]['M']
        range_pred = self.look_up[idx]['K']

        # Make these lists
        past_o_g = torch.stack([a for a in self.data[task][range_past, 0]])
        past_adj = torch.stack([a for a in self.data[task][range_past, 1]])
        past_a = torch.stack([a for a in self.data[task][range_past[:-1], 2]])
        past_o_f = torch.stack([a for a in self.data[task][range_past, 3]])

        future_o_g = torch.stack([a for a in self.data[task][range_pred, 0]])
        future_adj = torch.stack([a for a in self.data[task][range_pred, 1]])
        future_a = torch.stack([a for a in self.data[task][range_pred[:-1], 2]])
        future_o_f = torch.stack([a for a in self.data[task][range_pred, 3]])

        params = torch.from_numpy(np.asarray([self.k[task], self.b[task]])).float()

        # datapoint = {'past': {'f': past_o_f, 'g': past_o_g, 'a': past_a, 'adj': past_adj},
        #              'pred': {'f': future_o_f, 'g': future_o_g, 'a': future_a, 'adj': future_adj},
        #              }
        #
        # datapoint = [past_o_g,
        #              past_adj,
        #              past_a,
        #              past_o_f,
        #              future_o_g,
        #              future_adj,
        #              future_a,
        #              future_o_f
        #              ]

        return past_o_f, past_o_g, past_a, past_adj, future_o_f, future_o_g, future_a, future_adj, params

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
         return self.num_datapoints


if __name__ == '__main__':
    path = '../examples/data/env*'
    paths = glob.glob(path)

    dataset = PointcloudDataset(paths)