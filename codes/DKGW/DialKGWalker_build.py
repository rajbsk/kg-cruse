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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from logger import Logger
from tqdm import tqdm
from time import time

from DialKgWalker_model import SelfAttentionLayer, ModalityAttentionLayer, SentenceEncoder, DialogueEncoder, KGPathWalker

class KGPathWalkerModel(nn.Module):
    def __init__(self, opt):
        super(KGPathWalkerModel, self).__init__()
        self.n_entity = opt["n_entity"]
        self.n_relation = opt["n_relation"]
        self.batch_size = opt["batch_size"]
        self.lr = opt["lr"]
        self.word_dim = opt["word_dim"]
        self.hidden_dim = opt["hidden_dim"]
        self.epochs = opt["epochs"]
        self.device = opt["device"]
        self.n_hops = opt["n_hops"]
        self.model_name = opt["model_name"]
        self.model_directory = opt["model_directory"]

        self.entity_features = opt["entity_embeddings"].to(self.device)
        self.relation_features = opt["relation_embeddings"].to(self.device)
        self.word_features = opt["word_embeddings"].to(self.device)

        self.entity_features = nn.Embedding.from_pretrained(self.entity_features)
        self.relation_features = nn.Embedding.from_pretrained(self.relation_features)
        self.word_features = nn.Embedding.from_pretrained(self.word_features)

        self.word_embeddings = nn.Embedding(400001, 300)
        self.entity_embeddings = nn.Embedding(self.n_entity, 128)
        self.relation_embeddings = nn.Embedding(self.n_relation, 128)
        
        self.word_embeddings = self.word_embeddings.to(self.device)
        self.entity_embeddings = self.entity_embeddings.to(self.device)
        self.relation_embeddings = self.relation_embeddings.to(self.device)

        self.utterance_encoder = SentenceEncoder(opt).to(self.device)
        self.dialogue_encoder = DialogueEncoder(opt).to(self.device)
        self.modality_attention = ModalityAttentionLayer(opt).to(self.device)
        self.walker = KGPathWalker(opt).to(self.device)

        self.wf = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.wf = self.wf.to(self.device)

        self.optimizer = torch.optim.Adagrad( filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, lr_decay=0, eps=1e-8)
    
    def get_batch_data(self, batch):
        dialogue_padded, current_padded, dialogue_len, current_len, dialogue_mask, current_mask = batch[0], batch[1], batch[2], batch[3], batch[4],batch[5]
        entity_paths, relation_paths, seed_entities, true_entities, start_entities = batch[6], batch[7], batch[8], batch[9], batch[10]

        dialogue_padded = [dialogue.to(self.device) for dialogue in dialogue_padded]
        current_padded = current_padded.to(self.device)

        dialogue_len = [d_len.to(self.device) for d_len in dialogue_len]
        current_len = current_len.to(self.device)

        dialogue_mask = [mask.to(self.device) for mask in dialogue_mask]
        current_mask = current_mask.to(self.device)

        entity_paths = entity_paths.to(self.device)
        relation_paths = relation_paths.to(self.device)
        seed_entities = seed_entities.to(self.device)
        true_entities = true_entities.to(self.device)

        return dialogue_padded, current_padded, dialogue_len, current_len, dialogue_mask, current_mask, entity_paths, relation_paths, seed_entities, true_entities, start_entities

    def generate_negative_samples(self, current_batch_size):
        negative_entity_samples = (torch.Tensor(current_batch_size, 20).random_(0, self.n_entity-1)).long()
        negative_relation_samples = (torch.Tensor(current_batch_size, 20).random_(0, self.n_relation-1)).long()
        
        negative_entity_samples = negative_entity_samples.to(self.device)
        negative_relation_samples = negative_relation_samples.to(self.device)
        return negative_entity_samples, negative_relation_samples
    
    def loss_calc(self, x_bar, y_true, negative_samples): #(BxD), (BxD), (BxNxD) => int
        y_true = y_true.unsqueeze(1)
        
        x_bar = x_bar.unsqueeze(1)

        rhs = (negative_samples-y_true)
        rhs = x_bar*rhs
        rhs = torch.sum(rhs, dim=-1)

        lhs = negative_samples*y_true
        lhs = torch.sum(lhs, dim=-1)

        loss = lhs + rhs
        mask = loss<0
        loss[mask] = 0
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss

    def process_batch(self, batch, train=False):
        dialogue_padded, current, dialogue_len, current_len, dialogue_mask, current_mask, entity_paths, relation_paths, seed_entities, true_entities, _ = self.get_batch_data(batch)
        current_batch_size = len(current_len)

        dialogue_embed = [self.word_embeddings(dialogue) for dialogue in dialogue_padded]
        current = self.word_embeddings(current)

        dialogue_packed = []
        for i in range(len(dialogue_embed)):
            d_packed = pack_padded_sequence(dialogue_embed[i], dialogue_len[i].to("cpu"), batch_first=True, enforce_sorted=False)
            dialogue_packed.append(d_packed)
        current = pack_padded_sequence(current, current_len.to("cpu"), batch_first=True, enforce_sorted=False)

        current = self.utterance_encoder(current, current_mask)
        dialogue = self.dialogue_encoder(dialogue_packed, dialogue_mask)
        entity_embeddings = self.entity_embeddings(seed_entities)

        x_bar = self.modality_attention(entity_embeddings, current, dialogue)
        negative_entity_samples, negative_relation_samples = self.generate_negative_samples(current_batch_size)

        h_t = torch.zeros(current_batch_size, self.hidden_dim).to(self.device)
        c_t = torch.zeros(current_batch_size, self.hidden_dim).to(self.device)
        x_f = self.wf(x_bar)

        true_entities = self.entity_embeddings(true_entities)
        true_entity_path = self.entity_embeddings(entity_paths)
        true_relation_path = self.relation_embeddings(relation_paths)
        negative_entity = self.entity_embeddings(negative_entity_samples)
        negative_relation = self.relation_embeddings(negative_relation_samples)

        loss = self.loss_calc(x_f, true_entities, negative_entity)
        for i in range(self.n_hops):
            h_t, c_t, r_t = self.walker(x_bar, self.relation_embeddings.weight, h_t, c_t)
            entity_loss = self.loss_calc(h_t, true_entity_path[:, i, :], negative_entity)
            relation_loss = self.loss_calc(r_t, true_relation_path[:, i, :], negative_relation)

            loss += entity_loss + relation_loss

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()
            
    def train_model(self, trainDataLoader, devDataLoader):
        # logger = Logger("logs/")
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            dev_loss = 0
            cnt = 0
            for idx, batch in tqdm(enumerate(trainDataLoader)):
                batch_loss = self.process_batch(batch, train=True)
                train_loss += batch_loss
                cnt+=1
            
            train_loss = train_loss/cnt
            cnt = 0
            for idx, batch in tqdm(enumerate(devDataLoader)):
                batch_loss = self.process_batch(batch, train=False)
                dev_loss += batch_loss
                cnt+=1
            dev_loss = dev_loss/cnt

            print("Epoch = %d, Train Loss = %f, Dev Loss = %f"%(epoch+1, train_loss, dev_loss))
            if (epoch+1)%5==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))
            # logger.scalar_summary("Train Loss", train_loss, epoch+1)
            # logger.scalar_summary("Dev Loss", dev_loss, epoch+1)


