import pandas as pd 
import numpy as np 
import functools
import operator
from time import time
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertTokenizer, AlbertModel
import dgl
from dgl.sampling import sample_neighbors
from utils import load_pickle_file, _read_knowledge_graph_dialkg, _make_dgl_graph, _find_relation_entity_path

UNK = 0
PAD = 1
EOS=2

def pad_a_sequence(sequence):
    padded_sequence = pad_sequence(sequence, batch_first=True, padding_value=PAD)
    return padded_sequence

def dialkg_collate(batch):
    dialogue, dialogue_len, current, current_len  = [[] for _ in range(len(batch[0][0]))], [[] for _ in range(len(batch[0][0]))], [], []
    seed_entities = []
    entity_paths = []
    relation_paths = []
    true_entities = []
    start_entities = []
    
    for sample in batch:
        for i in range(len(sample[0])):
            dialogue[i].append(sample[0][i])
            dialogue_len[i].append(len(sample[0][i]))
        current.append(sample[1])
        current_len.append(len(sample[1]))
        seed_entities.append(sample[2])
        true_entities.append(sample[3])
        entity_paths.append(sample[4])
        relation_paths.append(sample[5])
        start_entities.append(sample[6])
    
    dialogue_padded = []
    dialogue_mask = []
    for i in range(len(dialogue)):
        d_padded = pad_a_sequence(dialogue[i])
        d_mask = d_padded!=PAD
        dialogue_padded.append(d_padded)
        dialogue_mask.append(d_mask)

    current = pad_a_sequence(current)
    current_mask = current!=PAD

    dialogue_len = [torch.tensor(length, dtype=torch.int64) for length in dialogue_len]
    current_len = torch.tensor(current_len, dtype=torch.int64)

    entity_paths = torch.stack(entity_paths)
    relation_paths = torch.stack(relation_paths)
    seed_entities = torch.tensor(seed_entities, dtype=torch.int64)
    true_entities = torch.tensor(true_entities, dtype=torch.int64)

    return [dialogue_padded, current,  dialogue_len, current_len, dialogue_mask, current_mask, entity_paths, relation_paths, seed_entities, true_entities, start_entities]


class DialKGDataset(Dataset):
    def __init__(self, opt, transform):
        self.transform = transform
        self.dataset = load_pickle_file(opt['dataset'])
        self.entity2entityId = load_pickle_file(opt['entity2entityId'])
        self.relation2relationId = load_pickle_file(opt['relation2relationId'])
        self.entity_embeddings = load_pickle_file(opt["entity_embeddings"])
        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])
        self.word_embeddings = load_pickle_file(opt["wordId2wordEmb"])
        self.heads, self.tails, self.relations = _read_knowledge_graph_dialkg(opt['knowledge_graph'], self.entity2entityId, self.relation2relationId)
        self.graph = _make_dgl_graph(self.heads, self.tails, self.relations)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __init__(self, opt):
        self.entity2entityId = load_pickle_file(opt['entity2entityId'])
        self.relation2relationId = load_pickle_file(opt['relation2relationId'])
        self.word2wordId = load_pickle_file(opt['word2wordId'])
        self.heads, self.tails, self.relations = _read_knowledge_graph_dialkg(opt['knowledge_graph'], self.entity2entityId, self.relation2relationId)
        self.graph = _make_dgl_graph(self.heads, self.tails, self.relations)
        self.n_hop = opt['n_hop']
        self.n_max = opt['n_max']
        self.max_dialogue_history = opt['max_dialogue_history']
        self.entity_eos = 0
        self.relation_eos = 0

    def sent2idx(self, sentence):
        idx_sent = []
        if len(sentence)==0:
            idx_sent = torch.tensor([EOS], dtype=torch.int64)
            return idx_sent

        for word in sentence.lower().split(" "):
            idx_sent.append(self.word2wordId[word])
        idx_sent.append(EOS)
        idx_sent = torch.tensor(idx_sent, dtype=torch.int64)
        return idx_sent

    def __call__(self, sample):
        current_utterance = sample[0]
        dialogue_history = sample[1]
        startEntities = sample[2]
        paths = sample[3]

        seed_entities = [self.entity2entityId[entity] for entity in startEntities]

        utterance = self.sent2idx(current_utterance)
        dialogue_history = dialogue_history[-3:]
        for _ in range(self.max_dialogue_history-len(dialogue_history)):
            dialogue_history = [""] + dialogue_history
        dialogue_history_idx = [self.sent2idx(sent) for sent in dialogue_history]
        
        entity_path = []
        relation_path = []

        for i in range(len(paths)):
            path = paths[i]
            if i==0:
                startEntities = path[0]
            entity_path.append(path[2])
            if path[1] == "~Author":
                path[1] = "~written_by"
            relation_path.append(path[1])
        
        true_response_entity = self.entity2entityId[entity_path[-1]]
        startEntities = self.entity2entityId[startEntities]
        entity_path = [self.entity2entityId[entity] for entity in entity_path] 
        relation_path = [self.relation2relationId[relation] for relation in relation_path]

        while(len(entity_path)!=self.n_hop):
            entity_path.append(entity_path[-1])
            relation_path.append(0)
        
        entity_path = torch.tensor(entity_path, dtype=torch.int64)
        relation_path = torch.tensor(relation_path, dtype=torch.int64)

        return [dialogue_history_idx, utterance, startEntities, true_response_entity, entity_path, relation_path, seed_entities]