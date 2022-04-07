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

def attnIO_collate(batch):
    dialogue_representations = []
    seed_entities = []
    subgraphs = []
    entity_paths = []
    for sample in batch:
        dialogue_representations.append(sample[0])
        seed_entities.append(sample[1])
        subgraphs.append(sample[2])
        entity_paths.append(sample[3])

    return [dialogue_representations, seed_entities, subgraphs, entity_paths]


class AttnIODataset(Dataset):

    def __init__(self, opt, transform):
        self.transform = transform
        self.dataset = load_pickle_file(opt['dataset'])
        self.entity2entityId = load_pickle_file(opt['entity2entityId'])
        self.relation2relationId = load_pickle_file(opt['relation2relationId'])
        self.dialogue2dialogueId = load_pickle_file(opt['dialogue2dialogueId'])
        self.dialogueId2AlbertRep = load_pickle_file(opt['dialogueId2AlbertRep'])
        self.entity_embeddings = load_pickle_file(opt["entity_embeddings"])
        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])

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
        self.dialogue2dialogueId = load_pickle_file(opt['dialogue2dialogueId'])
        self.dialogueId2AlbertRep = load_pickle_file(opt['dialogueId2AlbertRep'])
        self.heads, self.tails, self.relations = _read_knowledge_graph_dialkg(opt['knowledge_graph'], self.entity2entityId, self.relation2relationId)
        self.graph = _make_dgl_graph(self.heads, self.tails, self.relations)
        self.self_loop_id = self.relation2relationId['self loop']
        self.n_hop = opt['n_hop']
        self.n_max = opt['n_max']
        self.max_dialogue_history = opt['max_dialogue_history']

    def __call__(self, sample):
        current_utterance = sample[0]
        dialogue_history = sample[1]
        startEntities = sample[2]
        paths = sample[3]

        paths = torch.tensor([[self.entity2entityId[path[0]], self.relation2relationId[path[1]], self.entity2entityId[path[2]]] for path in paths])
        entity_path = _find_relation_entity_path(paths, self_loop_id=(self.self_loop_id))
        entity_path = [path[1].item() for path in entity_path]

        dialogue_context = ' '.join(dialogue_history[-(self.max_dialogue_history - 1):] + [current_utterance])
        dialogueId = self.dialogue2dialogueId[dialogue_context]
        dialogue_representation = self.dialogueId2AlbertRep[dialogueId]
        dialogue_representation = dialogue_representation.unsqueeze(0)

        seed_entities = [self.entity2entityId[entity] for entity in startEntities]
        source_entities = [self.entity2entityId[entity] for entity in startEntities]
        head_entities = []
        tail_entities = []
        edge_relations = []

        for _ in range(self.n_hop):
            subgraph = sample_neighbors(g=(self.graph), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
            edges = subgraph.edges()
            head_entities.extend(edges[0].tolist())
            tail_entities.extend(edges[1].tolist())
            edge_relations.extend(subgraph.edata['edge_type'].tolist())
            source_entities = list(set(edges[1].tolist()))

        edge_presence = defaultdict(int)
        for i in range(len(head_entities)):
            label = str(head_entities[i]) + '_' + str(tail_entities[i]) + '_' + str(edge_relations[i])
            edge_presence[label] += 1

        head_entities, tail_entities, edge_relations = [], [], []
        for key, value in edge_presence.items():
            head, tail, relation = key.split('_')
            head = int(head)
            tail = int(tail)
            relation = int(relation)
            head_entities.append(head)
            tail_entities.append(tail)
            edge_relations.append(relation)

        entities = head_entities + tail_entities + entity_path
        entities = list(set(entities))
        node2nodeId = defaultdict(int)
        nodeId2node = defaultdict(int)
        idx = 0
        for entity in entities:
            node2nodeId[entity] = idx
            nodeId2node[idx] = entity
            idx += 1

        indexed_head_entities = [node2nodeId[head_entity] for head_entity in head_entities]
        indexed_tail_entities = [node2nodeId[tail_entity] for tail_entity in tail_entities]

        entity_paths = [node2nodeId[node] for node in entity_path]
        seed_entities = [node2nodeId[node] for node in seed_entities]

        paths = [[node2nodeId[path[0].item()], path[1].item(), node2nodeId[path[2].item()]] for path in paths]

        indexed_head_entities = torch.tensor(indexed_head_entities, dtype=(torch.int64))
        indexed_tail_entities = torch.tensor(indexed_tail_entities, dtype=(torch.int64))
        edge_relations = torch.tensor(edge_relations, dtype=(torch.int64))

        subgraph = dgl.graph((torch.tensor([], dtype=(torch.int64)), torch.tensor([], dtype=(torch.int64))))
        subgraph.add_edges(indexed_head_entities, indexed_tail_entities, {'edge_type': edge_relations})
        subgraph = dgl.remove_self_loop(subgraph)
        subgraph_nodes = subgraph.nodes().tolist()
        subgraph_node_ids = [nodeId2node[node] for node in subgraph_nodes]
        subgraph_node_ids = torch.tensor(subgraph_node_ids, dtype=(torch.int64))
        subgraph.ndata['nodeId'] = subgraph_node_ids

        seed_entities = torch.tensor(seed_entities, dtype=(torch.int64))
        entity_paths = torch.tensor(entity_paths, dtype=(torch.int64))
        edges = subgraph.edges()
        heads, relations, tails = edges[0].tolist(), edge_relations.tolist(), edges[1].tolist()
        for path in paths:
            flag = 0
            for j in range(len(heads)):
                if path[0] == heads[j] and path[1] == relations[j] and path[2] == tails[j]:
                    flag = 1

            if not flag:
                subgraph.add_edge(u=(torch.tensor([path[0]])), v=(torch.tensor([path[2]])), data={'edge_type': torch.tensor([path[1]])})

        return [dialogue_representation, seed_entities, subgraph, entity_paths]