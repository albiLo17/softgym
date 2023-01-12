import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
import torch
import numpy as np
import random
from tqdm import tqdm
import glob

from logger import wandb_logger
from dataloader import PointcloudDataset

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
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()



np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # run_name = utils_main.get_run_name(args)
    run_name = 'test'
    print(f"Experiment :{run_name}")

    # writer = wandb_logger(args, run_name=run_name)

    # access dataset
    path = '../examples/data/env*'
    paths = glob.glob(path)
    paths.sort()

    dataset_train, dataset_test, dataset_val = PointcloudDataset(paths[:600]), PointcloudDataset(paths[600:800]), PointcloudDataset(paths[800:])

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0)

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