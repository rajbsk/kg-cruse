import math
from collections import defaultdict, deque

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from sklearn.metrics import roc_auc_score

from logger import Logger
from tqdm import tqdm
from time import time as tme

from kg_env import BatchKGEnvironment
from model import Model
from utils import load_pickle_file

class ConvKGTrainer(nn.Module):
    def __init__(self, opt):
        super(ConvKGTrainer, self).__init__()
        self.opt = opt
        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]

        self.device = opt["device"]
        self.batch_size = opt["batch_size"]
        self.n_entity = opt["n_entity"]
        self.n_relation = opt["n_relation"]
        self.epochs = opt["epochs"]
        self.entity_dim = opt["entity_dim"]
        self.relation_dim = opt["relation_dim"]
        self.lr = opt["lr"]
        
        self.entity_embedding = opt["entity_embeddings"].to(self.device)
        self.relation_embedding = opt["relation_embeddings"].to(self.device)
        self.entity_embedding.requires_grad = True
        self.relation_embedding.requires_grad = True
        self.graph = opt["graph"]

        self.env = BatchKGEnvironment(self.graph, self.opt)
        self.model = Model(self.opt)
        self.model.to(self.device)
        self.actor_optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.model.actor.parameters()), self.lr, weight_decay=1e-3)
    
    def log_params(self, logger, params, epoch):
        for tag, value in params:
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            if value.grad != None:
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
        
        if epoch%10==0:
            torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))

    def train_model(self, trainDataLoader):
        # logger = Logger("logs/")
        total_losses, total_plosses = [], []
        
        self.model.train()
        step = 0
        for epoch in range(self.epochs):
            for batch in tqdm(trainDataLoader):
                # Start of episodes
                batch_dialogue_history, batch_entity_history, batch_true_path = batch[0], batch[1], batch[2]
                batch_dialogue_history = batch_dialogue_history.to(self.device)
                batch_true_path = torch.tensor(batch_true_path, dtype=torch.int64)
                batch_state = self.env.reset(batch_entity_history, batch_dialogue_history)

                done = False
                time = 0
                while not done:
                    batch_dialogue_history, batch_entity_history, batch_path = batch_state
                    batch_act_mask = self.env.batch_action_mask()

                    batch_act_idx = self.model.select_action(batch_path, batch_dialogue_history, self.env._batch_curr_actions,
                                                            batch_act_mask, batch_true_path[:, :time, :], batch_true_path[:, time, :], time)
                    
                    batch_state, _, done = self.env.batch_step(batch_act_idx, [path[time, 1] for path in batch_true_path], time)

                    time += 1
                # End of episodes

                # update policy
                loss, ploss = self.model.update()
                
                total_losses.append(loss)
                total_plosses.append(ploss)
                step += 1

                if step%100==0 and step>0:
                    avg_loss = np.mean(total_losses)
                    avg_ploss = np.mean(total_plosses)
                    total_losses, total_plosses = [], []
                    # logger.scalar_summary("Loss Loss", avg_loss, step)
                    # logger.scalar_summary("Ploss Loss", avg_ploss, step)

            if (epoch+1)%20 == 0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))


