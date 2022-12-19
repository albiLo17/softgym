import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
import torch
import numpy as np
import random

from arguments import get_argparse
from logger import wandb_logger
from dataloaders.dataset import DeformableDataset

from torch.utils.data import DataLoader

# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

# Add needed environmental paths
import os
os.environ['PYFLEXROOT'] = os.environ['PWD'] + "/PyFlex"
os.environ['LD_LIBRARY_PATH'] = os.environ['PYFLEXROOT'] + "/external/SDL2-2.0.4/lib/x64"


import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Process some integers.')
# ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
parser.add_argument('--env_name', type=str, default='ClothDrag')
parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
parser.add_argument('--num_variations', type=int, default=3, help='Number of environment variations to be generated')
parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
parser.add_argument('--save_data', type=bool, default=True, help='save trajectory in a folder')

args = parser.parse_args()



np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    run_name = utils_main.get_run_name(args)
    print(f"Experiment :{run_name}")

    writer = wandb_logger(args, run_name=run_name)

    dataset_train, dataset_test, dataset_val = utils_main.load_datasets(args, shuffle=True)

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Needed to parallelize neighbors
    masks = utils_main.get_masks(dataset_train.adjs[0], order=args.adj_order)
    masks = [torch.from_numpy(m).float().to(device) for m in masks]

    # Get the specific network depending on the type of adaptation required
    model = OnlineAdaptModel(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    trainer = Trainer(args=args,
                      opt=opt,
                      device=device,
                      model=model,
                      dataloaders=[train_dataloader,
                                   val_dataloader,
                                   test_dataloader],
                      writer=writer)

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = trainer.train()
        train_losses.extend([train_loss])

        test_loss = trainer.validation()
        test_losses.extend([test_loss])

        # TODO: log here

        # print()

    # print()