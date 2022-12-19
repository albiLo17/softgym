import utils
import torch.nn as nn
import torch
import copy
import numpy as np
import utils_main
import random

class Trainer():

    def __init__(self, args, opt, device, model, dataloaders, writer):
        super(Trainer).__init__()

        self.args = args
        self.device = device

        self.method = args.method

        self.opt = opt
        self.mse = nn.MSELoss()
        self.model = model

        self.writer = writer

        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders


    def forward(self, dataloader, train=False):
        tot_force_loss = 0
        tot_graph_loss = 0
        tot_loss = 0

        for ix, batch in enumerate(dataloader):

            # past_o_g = batch[0].to(self.device)
            # past_adj = batch[1].to(self.device)
            # past_a = batch[2].to(self.device)
            # past_o_f = batch[3].to(self.device)
            # future_o_g = batch[4].to(self.device)
            # future_adj = batch[5].to(self.device)
            # future_a = batch[6].to(self.device)
            # future_o_f = batch[7].to(self.device)

            # f, g, a, adj
            past_obs = [b.to(self.device) for b in batch[:4]]
            future_obs = [b.to(self.device) for b in batch[4:8]]
            params = batch[8].to(self.device)

            pred_forces, pred_graphs, z = self.model(past_obs, future_obs, params)

            force_loss = self._get_loss(pred=pred_forces, gt=future_obs[0][:, 1:], type='force')  #TODO: check indeces
            graph_loss = self._get_loss(pred=pred_graphs, gt=future_obs[1][:, 1:], type='graph')
            loss = force_loss + graph_loss

            z_loss = 0
            if self.method == 3:
                z_loss = self._get_loss(pred=z[0], gt=z[1], type='z')
                loss += z_loss


            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            tot_force_loss += force_loss.detach().cpu().item()
            tot_graph_loss += graph_loss.detach().cpu().item()
            tot_loss += loss.detach().cpu().item()

        return tot_loss, tot_force_loss, tot_graph_loss, z_loss



    def train(self,):
        self.model.train()

        tot_loss, tot_force_loss, tot_graph_loss, z_loss = self.forward(dataloader=self.train_dataloader,
                                                                train=True
                                                                )


        self.writer.add_loss({'training - TOT loss': tot_loss,
                     'training - F loss': tot_force_loss,
                     'training - G loss': tot_graph_loss
                     })

        if self.method == 3:
            self.writer.add_loss({'training - z loss': z_loss,
                                  })


        return tot_loss

    def validation(self,):
        self.model.eval()
        with torch.no_grad():
            tot_loss, tot_force_loss, tot_graph_loss, z_loss = self.forward(dataloader=self.val_dataloader,
                                                                    train=False
                                                                    )

            self.writer.add_loss({'test - TOT loss': tot_loss,
                         'test - F loss': tot_force_loss,
                         'test - G loss': tot_graph_loss
            })

            if self.method == 3:
                self.writer.add_loss({'test - z loss': z_loss,
                                      })

            return tot_loss

    def _get_loss(self, pred, gt, type='graph'):
        if type == 'graph':
            return utils.get_graph_loss(pred, gt, loss_type='euc')

        # Loss for forces and z
        return self.mse(target=gt, input=pred)


