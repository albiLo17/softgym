import os
import matplotlib.pyplot as plt
import os
from matplotlib import collections
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# WandB – Import the wandb library
import wandb

import torch
import numpy as np

class wandb_logger():
    def __init__(self, args, run_name):
        self.out_dir = args.outdir
        self.plot_figures = args.plot_figures
        self.best_test_loss = None

        self.run_name = run_name
        self.project = args.project

        # Init wandb
        wandb.init(entity="albilo", project=self.project, name=run_name)
        wandb.config.update(args)

        # Folders
        self.check_point_dir = args.outdir + 'checkpoint/' + args.project  + '/'

        self.plot_img_freq = args.plot_frequency

        self.figures_dir = os.path.join(self.check_point_dir + self.run_name + '/', 'figures/')
        self.loss_dir = os.path.join(self.check_point_dir + self.run_name + '/', 'losses/')
        self.results_dir = os.path.join(self.check_point_dir + self.run_name + '/', 'results/')
        self.models_dir = os.path.join(self.check_point_dir + self.run_name + '/', '')

        # logdir = args.outdir + 'logs/' + args.exp_name + '/' + self.run_name

        self.check_folders()

    def check_folders(self):
        if not os.path.exists(self.check_point_dir):
            utils_main.make_dir(self.check_point_dir)
        if self.plot_figures:
            if not os.path.exists(self.figures_dir):
                utils_main.make_dir(self.figures_dir)
        if not os.path.exists(self.loss_dir):
            utils_main.make_dir(self.loss_dir)
        if not os.path.exists(self.results_dir):
            utils_main.make_dir(self.results_dir)

    def add_loss(self, losses):
        wandb.log(losses)