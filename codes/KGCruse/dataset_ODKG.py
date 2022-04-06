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
from time import time as tme

import dgl
from dgl.sampling import sample_neighbors

from utils import load_pickle_file, _read_knowledge_graph_dialkg, _make_dgl_graph, _find_entity_path, _find_relation_entity_path, _load_kg_embeddings

def ConvKG_collate(batch):
    dialogue_representations = []
    seed_entities = []
    paths = []
    for sample in batch:
        dialogue_representations.append(sample[0])
        seed_entities.append(sample[1])
        paths.append(sample[2])

    dialogue_representations = torch.stack(dialogue_representations)
    paths = torch.tensor(paths)
    return dialogue_representations, seed_entities, paths 

class ConvKGDataset(Dataset):
    def __init__(self, opt, transform):
        self.transform = transform
        self.entity2entityId = load_pickle_file(opt["entity2entityId"])
        self.relation2relationId = load_pickle_file(opt["relation2relationId"])
        self.dataset = load_pickle_file(opt["dataset"])
        self.entity_embeddings = load_pickle_file(opt["entity_embeddings"])
        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])

        self.heads, self.tails, self.relations = _read_knowledge_graph_dialkg(opt["knowledge_graph"], self.entity2entityId, self.relation2relationId)
        self.graph = _make_dgl_graph(self.heads, self.tails, self.relations)
        if opt["test"]:
            self.graph = dgl.transform.remove_self_loop(self.graph)
        self.self_loop_id = self.relation2relationId["self loop"]

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
    """Convert the entities to indexes.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, opt):
        self.device = opt["device"]
        self.entity2entityId = load_pickle_file(opt["entity2entityId"])
        self.relation2relationId = load_pickle_file(opt["relation2relationId"])
        self.self_loop_id = self.relation2relationId["self loop"]
        self.dialogue2dialogueId = load_pickle_file(opt["dialogue2dialogueId"])
        self.dialogueId2AlbertRep = load_pickle_file(opt["dialogueId2AlbertRep"])
        self.max_dialogue_history = opt["max_dialogue_history"]
        

    def __call__(self, sample):
        current_utterance = sample[0]
        dialogue_history = sample[1]
        startEntities = sample[2]
        paths_named = sample[3]
        
        paths = [[self.entity2entityId[path[0]], self.relation2relationId[path[1]], self.entity2entityId[path[2]]] for path in paths_named]

        relation_entity_path = _find_relation_entity_path(paths, self_loop_id=self.self_loop_id)

        dialogue_context = " ".join(dialogue_history[-(self.max_dialogue_history-1):] + [current_utterance])
        dialogueId = self.dialogue2dialogueId[dialogue_context]
        dialogue_representation = self.dialogueId2AlbertRep[dialogueId]

        seed_entities = [self.entity2entityId[entity] for entity in startEntities]
        return [dialogue_representation, seed_entities, relation_entity_path]
