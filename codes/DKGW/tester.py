from __future__ import absolute_import, division, print_function

import sys
import os
split_id = sys.argv[1]
# gpu_id = "4"
# split = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from math import log
from datetime import datetime
from tqdm import tqdm
import math
import itertools
from torchvision import transforms
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl

import threading
from functools import reduce

from dataset import DialKGDataset, ToTensor, dialkg_collate
from DialKGWalker_build import KGPathWalkerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def get_neighbors(graph, source):
    subgraph = dgl.sampling.sample_neighbors(graph, source, -1, edge_dir = "out")
    neighbor_relations, neighbor_entities = subgraph.edata["edge_type"], subgraph.edges()[1]
    neighbor_relations = neighbor_relations.tolist() + [0]
    neighbor_relations = torch.tensor(neighbor_relations, dtype=torch.int64)
    neighbor_entities = neighbor_entities.tolist() + [source]
    neighbor_entities = torch.tensor(neighbor_entities, dtype=torch.int64)

    return neighbor_relations, neighbor_entities

recall_entity = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
recall_path = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
recall_relation = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}

def fuse_edges(probs_pool, path_pool):
    path_probs = defaultdict(list)
    for i in range(len(path_pool)):
        path = path_pool[i]
        if 0 in path:
            continue
        probs = probs_pool[i]
        path_tuple = (path[0], path[1], path[2])
        path_probs[path_tuple].append(probs)
    
    for path, probs in path_probs.items():
        p1, p2, p3 = 0, 0, 0
        for prob in probs:
            p1 = prob[0]
            p2 += prob[1]
            p3 += prob[2]
        path_probs[path] = [p1, p2, p3]

    path_pool, probs_pool = [], []
    for path in path_probs:
        path_pool.append(path)
        probs_pool.append(path_probs[path])
    return probs_pool, path_pool

def filter_paths(paths):
    last_entities = set()
    final_paths = []
    for path in paths:
        if path[-1] not in last_entities:
            last_entities.add(path[-1])
            final_paths.append(path)
    return final_paths

def remove_incorrect_paths(paths):
    final_paths = []
    for path in paths:
        if 0 not in path:
            final_paths.append(path)
    return final_paths

def calculate_metrics(paths, probs, true_path):
    K = [1, 3, 5, 10, 25]
    paths = remove_incorrect_paths(paths)
    filtered_paths = filter_paths(paths)
    
    entities = [path[2] for path in filtered_paths]
    recall_entity["counts"] += 1
    recall_path["counts"] += 1

    # gt = [entityId2entity[entityId] for entityId in true_path]
    x = 0
    for k in K:
        if true_path in paths[:k]:
            recall_path["counts@"+str(k)] += 1
        if true_path[2] in entities[:k]:
            x = 1
            recall_entity["counts@"+str(k)] += 1
    # if not x:
    #     dialogue = dialogueId2dialogue[dialogueId[0]]
    #     entity_paths = []
    #     for path in paths[:k]:
    #         entity_path = []
    #         for entityId in path:
    #             entity_path.append(entityId2entity[entityId])
        #     entity_paths.append(entity_path)
        # test = 1

def get_ent_scores(h_t, r_t, n_ent_emb, n_rel_emb):
    lhs = (h_t*n_ent_emb)
    lhs = torch.sum(lhs, dim=-1)
    rhs = (r_t*n_rel_emb)
    rhs = torch.sum(rhs, dim=-1)

    scores = (lhs + rhs)
    scores =  F.softmax(scores)
    return scores

def batch_beam_search(model, batch, device, topk, opt, graph):
    dialogue_padded, current, dialogue_len, current_len, dialogue_mask, current_mask, entity_paths, relation_paths, seed_entities, true_entities, start_entities = model.get_batch_data(batch)
    current_batch_size = len(current_len)
    dialogue_embed = [model.word_embeddings(dialogue) for dialogue in dialogue_padded]
    current = model.word_embeddings(current)

    dialogue_packed = []
    for i in range(len(dialogue_embed)):
        d_packed = pack_padded_sequence(dialogue_embed[i], dialogue_len[i], batch_first=True, enforce_sorted=False)
        dialogue_packed.append(d_packed)
    current = pack_padded_sequence(current, current_len, batch_first=True, enforce_sorted=False)

    current = model.utterance_encoder(current, current_mask)
    dialogue = model.dialogue_encoder(dialogue_packed, dialogue_mask)
    entity_embeddings = model.entity_embeddings(seed_entities)

    x_bar = model.modality_attention(entity_embeddings, current, dialogue)

    h_t = torch.zeros(current_batch_size, model.hidden_dim).to(device)
    c_t = torch.zeros(current_batch_size, model.hidden_dim).to(device)
    start_entities = start_entities[0]
    start_entities = [[ent] for ent in start_entities]
    path_pool = start_entities[:]
    probs_pool = [[1] for _ in range(len(start_entities))]
    hops = model.n_hops-1
    for i in range(hops):
        new_paths_pool, new_probs_pool = [], []
        for j in range(len(path_pool)):
            path = path_pool[j]
            source = path[-1]
            source_ent = torch.tensor([path[0]], dtype=torch.int64).to(device)
            source_ent_emb = model.entity_embeddings(source_ent)
            x_bar = model.modality_attention(source_ent_emb, current, dialogue)
            # x_bar = x_bar.unsqueeze(1)
            h_t = torch.zeros(current_batch_size, model.hidden_dim).to(device)
            c_t = torch.zeros(current_batch_size, model.hidden_dim).to(device)

            source = path_pool[j][-1]
            current_path = path_pool[j]
            current_prob = probs_pool[j]
            for _ in range(len(path)):
                h_t, c_t, r_t = model.walker(x_bar, model.relation_embeddings.weight, h_t, c_t)

            neighbor_relations, neighbor_entities = get_neighbors(graph, source)
            neighbor_relations = neighbor_relations.to(device)
            neighbor_entities = neighbor_entities.to(device)
            n_rel_emb = model.relation_embeddings(neighbor_relations)
            n_ent_emb = model.entity_embeddings(neighbor_entities)
            ent_scores = get_ent_scores(h_t, r_t, n_ent_emb, n_rel_emb)
            tk = min(len(n_ent_emb), topk[i])
            top_ent_scores, top_ent_idxs = torch.topk(ent_scores, k=tk)
            top_entities = neighbor_entities[top_ent_idxs].tolist()
            top_ent_scores = top_ent_scores.tolist()
            for k in range(tk):
                new_paths_pool.append(current_path[:]+[top_entities[k]])
                new_probs_pool.append(current_prob[:]+[top_ent_scores[k]])
        path_pool = new_paths_pool[:]
        probs_pool = new_probs_pool[:]

    probs_pool = [reduce(lambda x, y: x*y, probs) for probs in probs_pool]
    probs_entity = zip(probs_pool, path_pool)
    probs_entity = sorted(probs_entity, key=lambda x:x[0], reverse=True)
    probs_entity = zip(*probs_entity)
    probs_entity = [list(a) for a in probs_entity]
    probs_pool , path_pool = probs_entity[0], probs_entity[1]
    entity_paths = entity_paths[0].to("cpu")
    seed_entities = seed_entities.to("cpu")
    true_path = [seed_entities.item()] + entity_paths.tolist()[:2]
    calculate_metrics(path_pool, probs_pool, true_path)


def predict_paths(policy_file, ConvKGDatasetLoaderTest, opt, graph):
    print('Predicting paths...')
    pretrain_sd = torch.load(policy_file)
    model = KGPathWalkerModel(opt).to(opt["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt["device"])

    K = [[2, 5, 5], [2, 10, 5], [2, 10, 10], [2, 15, 5], [2, 15, 10], [2, 15, 15], [2, 20, 5], [2, 20, 10], [2, 20, 15], [2, 20, 20], [2, 25, 5],
        [2, 25, 10], [2, 25, 15], [2, 25, 20], [2, 25, 25], [2, 30, 5], [2, 30, 10], [2, 30, 15], [2, 30, 15], [2, 30, 20], [2, 30, 25], [2, 30, 30],
        [2, 35, 5], [2, 35, 10], [2, 35, 15], [2, 35, 20], [2, 35, 25], [2, 35, 30], [2, 35, 35], [2, 40, 5], [2, 40, 10], [2, 40, 15], [2, 40, 20],
        [2, 40, 25], [2, 40, 30], [2, 40, 35], [2, 40, 40], [2, 45, 5], [2, 45, 10], [2, 45, 25], [2, 45, 45], [2, 50, 5], [2, 50, 10], [2, 50, 15],
        [2, 50, 20], [2, 50, 25], [2, 50, 50]]
    K = [[25, 25]]
    for ks in K:
        for batch in tqdm(ConvKGDatasetLoaderTest):
            batch_beam_search(model, batch, opt["device"], topk = ks, opt=opt, graph=graph)
        
        for k, v in recall_entity.items():
            if "@" in k:
                recall_entity[k] /= recall_entity["counts"]

        for k, v in recall_path.items():
            if "@" in k:
                recall_path[k] /= recall_path["counts"]
        


        print(ks)
        path_res = str(recall_path["counts@1"]*100) + "\t" + str(recall_path["counts@3"]*100) + "\t" + str(recall_path["counts@5"]*100) + "\t" + str(recall_path["counts@10"]*100) + "\t" + str(recall_path["counts@25"]*100) + "\t" + str(recall_path["counts"])
        entity_res = str(recall_entity["counts@1"]*100) + "\t" + str(recall_entity["counts@3"]*100) + "\t" + str(recall_entity["counts@5"]*100) + "\t" + str(recall_entity["counts@10"]*100) + "\t" + str(recall_entity["counts@25"]*100) + "\t" + str(recall_path["counts"])
        print(path_res)
        print(entity_res)
        print(recall_relation)

        for k in recall_entity.keys():
            recall_entity[k] = 0
        for k in recall_path.keys():
            recall_path[k] = 0
        for k in recall_relation.keys():
            recall_relation[k] = 0

if __name__ == '__main__':
    data_directory = "../../datasets/dataset_baseline/"
    splits_directory = "../../datasets/splits/split_"+split_id+"/"
    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "word2wordId": data_directory+"word2wordId.pkl", "wordId2wordEmb": data_directory+"wordId2wordEmb.pkl",
                    "dataset": splits_directory+"dataset_test.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 3, "n_max": 100, "max_dialogue_history": 3}
    
    DialKG_dataset_train = DialKGDataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))

    opt_model = {"n_entity": len(DialKG_dataset_train.entity2entityId)+1, "n_relation": len(DialKG_dataset_train.relation2relationId)+1,
                "entity2entityId": opt_dataset_train["entity2entityId"], "entity_embedding_path": opt_dataset_train["entity_embeddings"],
                "entity_embeddings": DialKG_dataset_train.entity_embeddings, "relation_embeddings": DialKG_dataset_train.relation_embeddings, "word_embeddings": DialKG_dataset_train.word_embeddings,
                "hidden_dim":128, "word_dim": 300, "batch_size":1, "device": device, "lr": 1e-2, "lr_reduction_factor":0.1,
                "epochs": 5, "n_hops": 3, "model_directory": "models/", "model_name": "DialKGWalker", "clip": 5}

    ConvKGDatasetLoaderTrain = DataLoader(DialKG_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=dialkg_collate)
    graph = DialKG_dataset_train.graph
    policy_file = opt_model["model_directory"] + "model_"+split_id+"_25"
    # path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    predict_paths(policy_file, ConvKGDatasetLoaderTrain, opt_model, graph)
