import math
from collections import defaultdict
import gc

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from sklearn.metrics import roc_auc_score
from parlai.utils.misc import round_sigfigs

from logger import Logger
from tqdm import tqdm
from time import time

from attnio_model_new import AttnIO


def _get_instance_path_loss(graph, path):
    epsilon = 1e-30
    # denominator = (graph.num_nodes()*epsilon+1)
    scores = []
    for i in range(len(path)):
        # node_time_scores = ((graph.ndata["a_"+str(i)])/denominator) + (epsilon/denominator)
        node_time_scores = graph.ndata["a_"+str(i)] + epsilon
        node_time = path[i]
        score = node_time_scores[node_time]
        scores.append(-torch.log(score))
    scores = torch.stack(scores)
    return scores.sum(-1)


class AttnIOModel(nn.Module):
    def __init__(self, opt):
        super(AttnIOModel, self).__init__()

        self.device = opt["device"]
        self.n_entity = opt["n_entity"]
        self.n_relation = opt["n_relation"]
        self.out_dim = opt["out_dim"]
        self.in_dim = opt["in_dim"]
        self.lr = opt["lr"]
        self.lr_reduction_factor = opt["lr_reduction_factor"]
        self.epochs = opt["epoch"]
        self.attn_heads = opt["attn_heads"]
        self.beam_size = opt["beam_size"]
        self.clip = opt["clip"]
        self.self_loop_id = opt["self_loop_id"]

        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]

        self.entity_emb = Embedding(self.n_entity, 768)
        self.entity_emb.weight.data.copy_(opt["entity_embeddings"])
        
        self.relation_emb = Embedding(self.n_relation, 768)
        self.relation_emb.weight.data.copy_(opt["relation_embeddings"])
        self.model = AttnIO(self.in_dim, self.out_dim, self.attn_heads, self.entity_emb, self.relation_emb, self.self_loop_id, self.device)

        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)

        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        self.reset_metrics()
    
    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0
    
    def report(self):
        m = {}
        # Top-k recommendation Recall
        for x in sorted(self.metrics):
            if x.startswith("recall"):
                m[x] = self.metrics[x] / self.counts[x]

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def forward(self, dialogue_representation, seed_entities, subgraph):
        subgraph = self.model(subgraph, seed_entities, dialogue_representation)
        return subgraph

    def get_batch_data(self, batch):
        dialogue_representations, seed_entities, subgraphs, paths = batch[0], batch[1], batch[2], batch[3]
        return dialogue_representations, seed_entities, subgraphs, paths

    def process_batch(self, batch, train=False):        
        dialogue_representations, seed_entities, subgraphs, paths = self.get_batch_data(batch)
        batch_size = len(batch[0])
        batch_loss = []
        updated_subgraphs = []
        for i in range(batch_size):
            dialogue_representation = dialogue_representations[i].to(self.device)
            seed_entity = seed_entities[i].to(self.device)
            subgraph = subgraphs[i].to(self.device)
            path = paths[i].to(self.device)
            updated_subgraph = self(dialogue_representation, seed_entity, subgraph)
            updated_subgraphs.append(updated_subgraph)
            instance_loss = _get_instance_path_loss(updated_subgraph, path)
            batch_loss.append(instance_loss)
        
        batch_loss = torch.stack(batch_loss).sum(-1)/batch_size
        
        # print(batch_loss.item())
        if train:
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()

        return batch_loss.item(), updated_subgraphs
    
    def train_model(self, trainDataLoader, devDataLoader):
        # logger = Logger("logs/")
        self.optimizer.zero_grad()
        ins=0
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            cnt = 0
            for idx, batch in tqdm(enumerate(trainDataLoader)):
                batch_loss, _ = self.process_batch(batch, train=True)
                train_loss += batch_loss
                cnt+=1
            
            train_loss = train_loss/cnt
            cnt = 0
            dev_loss = 0
            for idx, batch in tqdm(enumerate(devDataLoader)):
                batch_loss, _ = self.process_batch(batch, train=False)
                dev_loss += batch_loss
                cnt+=1
            dev_loss = dev_loss/cnt
            
            self.lr_scheduler.step(dev_loss)
            print("Epoch: %d, Train Loss: %f" %(epoch, train_loss))
            print("Epoch: %d, Dev Loss: %f" %(epoch, dev_loss))
            
            # Logging parameters
            p = list(self.named_parameters())
            # logger.scalar_summary("Train Loss", train_loss, ins+1)
            # logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            ins+=1
            if (epoch+1)%2==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))       

