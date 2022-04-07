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

from AttnIO_model import AttnIO
from utils import find_all_paths, get_neighbors


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

        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]

        self.entity_emb = Embedding(self.n_entity, 768)
        self.entity_emb.weight.data.copy_(opt["entity_embeddings"])
        self.relation_emb = Embedding(self.n_relation, 768)
        self.relation_emb.weight.data.copy_(opt["relation_embeddings"])
        self.model = AttnIO(self.in_dim, self.out_dim, self.attn_heads, self.entity_emb, self.relation_emb)

        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr, weight_decay=0.01)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)

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
        if train:
            self.optimizer.zero_grad()
        
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
            batch_loss.backward()
            self.optimizer.step()

        return batch_loss.item(), updated_subgraphs

    def eval_metrics(self, subgraphs, batch):
        _, seed_entities, _, paths = self.get_batch_data(batch)
        batch_size = len(batch[0])
        for i in range(batch_size):
            s_entities = seed_entities[i]
            head, tail = subgraphs[i].edges()
            heads = head.to("cpu").tolist()
            tails = tail.to("cpu").tolist()
            n_nodes = subgraphs[i].num_nodes()
            neighbors = get_neighbors(heads, tails)

            a_0 = subgraphs[i].ndata["a_0"].to("cpu")
            a_1 = subgraphs[i].ndata["a_1"].to("cpu")
            a_2 = subgraphs[i].ndata["a_2"].to("cpu")
            
            t_0_scores, t_0_nodes = torch.topk(a_0, min(self.beam_size, len(s_entities)))
            t_0_scores = t_0_scores.tolist()
            t_0_nodes = t_0_nodes.tolist()

            scores = []

            for a in range(len(t_0_nodes)):
                t_0_node_neighbors = neighbors[t_0_nodes[a]]
                mask = torch.zeros(n_nodes)
                mask[t_0_node_neighbors] = 1
                t_1_scores = a_1*mask

                t_1_scores, t_1_nodes = torch.topk(t_1_scores, min(self.beam_size, len(t_0_node_neighbors)))
                t_1_scores = t_1_scores.tolist()
                t_1_nodes = t_1_nodes.tolist()

                for j in range(len(t_1_nodes)):
                    t_1_node_neighbors = neighbors[t_1_nodes[j]]
                    mask = torch.zeros(n_nodes)
                    mask[t_1_node_neighbors] = 1
                    t_2_scores = a_2*mask
                    
                    t_2_scores, t_2_nodes = torch.topk(t_2_scores, min(self.beam_size, len(t_1_node_neighbors)))
                    t_2_scores = t_2_scores.tolist()
                    t_2_nodes = t_2_nodes.tolist()

                    for k in range(len(t_2_nodes)):
                        scores.append(["_".join([str(t_0_nodes[a]), str(t_1_nodes[j]), str(t_2_nodes[k])]), t_0_scores[a] + t_1_scores[j] + t_2_scores[k]])
            
            path_scores = sorted(scores, reverse=True, key=lambda v: v[1])
            path_scores = [path for path, score in path_scores]
            actual_path = paths[i].tolist()
            actual_path = [str(entity) for entity in actual_path]
            actual_path_score = '_'.join(actual_path)

            if actual_path_score in path_scores[:1]:
                self.metrics["recall@1"] += 1
            if actual_path_score in path_scores[:3]:
                self.metrics["recall@3"] += 1
            if actual_path_score in path_scores[:5]:
                self.metrics["recall@5"] += 1
            if actual_path_score in path_scores[:10]:
                self.metrics["recall@10"] += 1
            if actual_path_score in path_scores[:25]:
                self.metrics["recall@25"] += 1
            self.counts["recall@1"] += 1
            self.counts["recall@3"] += 1
            self.counts["recall@5"] += 1
            self.counts["recall@10"] += 1
            self.counts["recall@25"] += 1

    def evaluate_model(self, dataLoader):
        self.eval()
        self.model.eval()
        self.reset_metrics()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataLoader):
                batch_loss, subgraphs = self.process_batch(batch, train=False)
                self.eval_metrics(subgraphs, batch)
                total_loss += batch_loss
                del subgraphs
                gc.collect()
                torch.cuda.empty_cache()
        print(self.report())
        self.train()
        # print(total_loss)
        return total_loss       
    
    def train_model(self, trainDataLoader, train_size, devDataLoader, dev_size):
        logger = Logger("logs/")
        self.optimizer.zero_grad()
        ins=0
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for idx, batch in tqdm(enumerate(trainDataLoader)):
                batch_loss, _ = self.process_batch(batch, train=True)
                train_loss += batch_loss
            
            dev_loss = self.evaluate_model(devDataLoader)
            train_loss = train_loss/train_size
            dev_loss = dev_loss/dev_size
            self.lr_scheduler.step(dev_loss)
            print("Epoch: %d, Train Loss: %f" %(epoch, train_loss))
            print("Epoch: %d, Dev Loss: %f" %(epoch, dev_loss))
            
            # Logging parameters
            p = list(self.named_parameters())
            logger.scalar_summary("Train Loss", train_loss, ins+1)
            logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), ins+1)
                if value.grad != None:
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), ins+1)
            ins+=1
            if epoch%2==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))       
