from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import numpy as np
import itertools
import math

import dgl
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.transform import add_self_loop
from dgl.sampling import sample_neighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torchgan.losses import MinimaxDiscriminatorLoss

SavedAction = namedtuple('SavedAction', ['log_prob', 'action_values', 'batch_loss'])

def MinimaxDiscriminatorLoss(true_action_probs, pred_action_probs):
    loss = -torch.log(true_action_probs+1e-30) - torch.log(1-pred_action_probs+1e-30)
    return loss

class Actor(nn.Module):
    def __init__(self, opt):
        super(Actor, self).__init__()
        self.n_entity = opt["n_entity"]
        self.n_relation = opt["n_relation"]
        self.dialogue_dim = opt["dialogue_dim"]
        self.entity_dim = opt["entity_dim"]
        self.relation_dim = opt["relation_dim"]
        self.state_dim = opt["state_dim"]
        self.device = opt["device"]

        self.entity_features = opt["entity_embeddings"].to(self.device)
        self.relation_features = opt["relation_embeddings"].to(self.device)
        self.entity_features.requires_grad_(requires_grad=True)
        self.relation_features.requires_grad_(requires_grad=True)

        # print(self.entity_features.size())
        # self.entity_features = torch.nn.Embedding(self.entity_features.size()[0], self.entity_features.size()[1])
        # self.relation_features = torch.nn.Embedding(self.relation_features.size()[0], self.relation_features.size()[1])

        # self.entity_features = self.entity_features.weight
        # self.relation_features = self.relation_features.weight
        # self.entity_features.requires_grad_(requires_grad=True)
        # self.relation_features.requires_grad_(requires_grad=True)


        self.lstm = nn.LSTM(input_size=self.entity_dim+self.relation_dim, hidden_size=self.entity_dim+self.relation_dim, batch_first=True, num_layers=3)
        self.ldialogue = nn.Linear(self.dialogue_dim, self.entity_dim+self.relation_dim, bias=False)
        self.l1 = nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim+self.relation_dim, bias=True)
        self.l2 = nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim+self.relation_dim, bias=True)
        self.l3 = nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim+self.relation_dim, bias=True)
        self.lq1 = nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim+self.relation_dim, bias=True)
        self.lq2 = nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim+self.relation_dim, bias=True)
        
    def forward(self, entity_path_history, relation_path_history, entity_candidates, relation_candidates, dialogue_representations):
        entity_path_history_embedding = self.entity_features[entity_path_history]
        relation_path_history_embedding = self.relation_features[relation_path_history]
        relation_entity_path_history_embedding = torch.cat([entity_path_history_embedding, relation_path_history_embedding], dim=-1)

        dialogue_embedding = self.ldialogue(dialogue_representations).unsqueeze(1)
        s_representation = torch.cat([dialogue_embedding, relation_entity_path_history_embedding], dim=1)

        _, (s, _) = self.lstm(s_representation)
        s = s[-1]
        # x = s.unsqueeze(-1)
        s = F.relu(self.l1(s))
        x = self.l3(F.relu(self.l2(s)))
        x = x.unsqueeze(-1)
        q = self.lq1(F.relu(self.lq2(s))).unsqueeze(-1)

        action_entity_embeddings = self.entity_features[entity_candidates]
        action_relation_embeddings = self.relation_features[relation_candidates]

        action_representation = torch.cat([action_entity_embeddings, action_relation_embeddings], dim=-1)

        action_logits = torch.bmm(action_representation, x).squeeze(-1)/math.sqrt(self.entity_dim+self.relation_dim)
        quality_values = torch.bmm(action_representation, q).squeeze(-1)/math.sqrt(self.entity_dim+self.relation_dim)
        return action_logits, quality_values

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.n_entity = opt["n_entity"]
        self.n_relation = opt["n_relation"]
        self.dialogue_dim = opt["dialogue_dim"]
        self.entity_dim = opt["entity_dim"]
        self.relation_dim = opt["relation_dim"]
        self.state_dim = opt["state_dim"]
        self.graph = opt["graph"]
        self.act_dim = self.graph.num_nodes()
        self.gamma = opt["gamma"]
        self.device = opt["device"]
        self.lr = opt["lr"]
        self.clip = opt["clip"]
        self.epochs = opt["epochs"]

        self.actor = Actor(opt)
        self.actor.to(self.device)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []
        self.discriminator_loss = []
        self.discriminator_reward = []
        self.critic_values = []

        self.actor_optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.actor.parameters()), self.lr, weight_decay=1e-3)
        self.DiscriminatorCriterion = MinimaxDiscriminatorLoss

        lr_lambda = lambda epoch: 1 - (epoch/self.epochs)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda)

        self.graph = self.graph.to(self.device)

    def select_action(self, path, dialogue_representations, batch_curr_actions, act_mask, true_path, true_acts, time_step):
        true_path = true_path.to(self.device)

        batch_relation_actions, batch_entity_actions = batch_curr_actions
        # relation_path_history, entity_path_history = torch.tensor(path[0], dtype=torch.int64).to(self.device), torch.tensor(path[1], dtype=torch.int64).to(self.device)
        true_relation_path_history, true_entity_path_history = true_path[:, :, 0], true_path[:, :, 1]

        batch_relation_actions = pad_sequence(batch_relation_actions, batch_first=True).to(self.device)
        batch_entity_actions = pad_sequence(batch_entity_actions, batch_first=True).to(self.device)

        relation_actions, entity_actions= batch_curr_actions
        act_mask = torch.tensor(act_mask).to(self.device)

        actor_logits, action_qualities = self.actor(true_entity_path_history, true_relation_path_history, batch_entity_actions, batch_relation_actions, dialogue_representations)
        actor_logits[~act_mask] = -1e6
        actor_probs = F.softmax(actor_logits, dim=-1)
        lp = torch.log(actor_probs+1e-8)
        acts = true_acts[:]

        batch_size = len(batch_entity_actions)
        true_indexes = torch.zeros((batch_size, 1), dtype=torch.int64)
        true_indexes = true_indexes.to(self.device)
        for i in range(batch_size):
            true_action_ent = acts[i][1]
            true_action_rel = acts[i][0]
            path_history = true_entity_path_history[i]

            entity_action = batch_entity_actions[i]
            relation_action = batch_relation_actions[i]
            y = relation_action.cpu().tolist()
            true_index_ent = (entity_action == true_action_ent).nonzero(as_tuple=True)[0]
            true_index_rel = (relation_action == true_action_rel).nonzero(as_tuple=True)[0]

            true_index_ent = set(true_index_ent.cpu().tolist())
            true_index_rel = set(true_index_rel.cpu().tolist())

            try:
                true_index = list(true_index_ent.intersection(true_index_rel))[0]
            except:
                true_index = list(true_index_ent)[0]
            true_indexes[i] = true_index
        
        true_action_log_probs = 0
        for i in range(batch_size):
            true_action_log_probs += lp[i][true_indexes[i]]
        true_action_log_probs /= batch_size
        x=10

        self.saved_actions.append(SavedAction(lp, actor_probs, true_action_log_probs))
        # self.entropy.append(m.entropy())
        return true_indexes.cpu().tolist()

    def update(self):
       
        num_steps = len(self.saved_actions)

        actor_loss = 0
        for i in range(num_steps):
            time_batch_loss = self.saved_actions[i][-1]
            actor_loss += -time_batch_loss  # Tensor of [bs, ]
        actor_loss /= num_steps
        loss = actor_loss
        
        self.actor_optimizer.zero_grad()
        
        loss.backward()

        self.actor_optimizer.step()

        del self.saved_actions[:]

        return loss.item(), actor_loss.item()
