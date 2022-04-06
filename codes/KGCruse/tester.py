from __future__ import absolute_import, division, print_function

import sys
import os
# gpu_id = sys.argv[1]
# split_id = sys.argv[2]
# fil = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from trainer import ConvKGTrainer
from kg_env import BatchKGEnvironment
from model import Model
from utils import *
from dataset_ODKG import ConvKGDataset, ToTensor, ConvKG_collate


from math import log
from datetime import datetime
from tqdm import tqdm
import argparse
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

import threading
from functools import reduce


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

with open("../../datasets/dataset_kgdial_ac/entityId2entity.pkl", "rb") as f:
    entityId2entity = pickle.load(f)
with open("../../datasets/dataset_kgdial_ac/relationId2relation.pkl", "rb") as f:
    relationId2relation = pickle.load(f)
with open("../../datasets/dataset_kgdial_ac/dialogueId2dialogue.pkl", "rb") as f:
    dialogueId2dialogue = pickle.load(f)


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

    gt = [entityId2entity[entityId] for entityId in true_path]
    x = 0
    for k in K:
        if true_path in paths[:k]:
            recall_path["counts@"+str(k)] += 1
        if true_path[2] in entities[:k]:
            recall_entity["counts@"+str(k)] += 1
        if true_path not in paths[:k]:
            x = 1

    if x:
        entity_paths = []
        true_paths = []
        for path in paths[:k]:
            entity_path = []
            for entityId in path:
                entity_path.append(entityId2entity[entityId])
            entity_paths.append(entity_path)
        for entityId in true_path:
            true_paths.append(entityId2entity[entityId])
        test = 1

def relation_accuracy(new_relation_path_pool, new_probs_pool, true_relation_path):
    new_probs_pool = [reduce(lambda x, y: x*y, probs) for probs in new_probs_pool]
    probs_relation_entity = zip(new_probs_pool, new_relation_path_pool)
    probs_relation_entity = sorted(probs_relation_entity, key=lambda x:x[0], reverse=True)
    probs_relation_entity = zip(*probs_relation_entity)
    probs_relation_entity = [list(a) for a in probs_relation_entity]
    probs_pool , path_pool = probs_relation_entity[0], probs_relation_entity[1]

    if true_relation_path in path_pool[:1]:
        recall_relation["counts@1"] += 1
    recall_relation["counts"] += 1


def batch_beam_search(model, batch, device, topk=[5, 10, 1], opt={}):
    def batch_action_mask(batch_acts):
        """Return action masks of size [bs, actions]."""
        entity_candidates = batch_acts 
        entity_candidates = pad_sequence(entity_candidates, batch_first=True)
        batch_action_mask = entity_candidates != 0
        return batch_action_mask

    with torch.no_grad():
        batch_dialogue_history, batch_entity_history, batch_true_path = batch[0], batch[1], batch[2]
        true_entity_path = batch_true_path[:, :, 1].tolist()
        true_relation_path = batch_true_path[:, :, 0].tolist()
        env = BatchKGEnvironment(model.graph, opt)

        batch_state = env.reset(batch_entity_history, batch_dialogue_history)  # numpy of [bs, dim]
        batch_dialogue_history, batch_entity_history, batch_path = batch_state
        done = False
        time = 0
        
        dialogue_history = batch_dialogue_history.to(device).clone()
        path_pool = env._batch_path  # list of list, size=bs
        probs_pool = [[] for _ in range(len(path_pool[0]))]
        model.eval()
        for hop in range(3):
            if hop==0:
                batch_relation_actions, batch_entity_actions = env._batch_curr_actions
            else:
                batch_relation_actions, batch_entity_actions = env._batch_get_actions(path_pool, False)

            batch_relation_actions = pad_sequence(batch_relation_actions, batch_first=True).to(device)
            batch_entity_actions = pad_sequence(batch_entity_actions, batch_first=True).to(device)

            relation_path_history, entity_path_history = torch.tensor(path_pool[0], dtype=torch.int64).to(device), torch.tensor(path_pool[1], dtype=torch.int64).to(device)
            actor_logits, _ = model.model.actor(entity_path_history, relation_path_history, batch_entity_actions, batch_relation_actions, dialogue_history)
            action_mask = batch_entity_actions!=0
            actor_logits[~action_mask] = -1e6
            actor_probs = F.softmax(actor_logits, dim=-1)

            k = min(topk[hop], len(actor_probs[0]))
            topk_probs, topk_actions = torch.topk(actor_probs, k=k)

            dialogue_history = dialogue_history.clone()
            dialogue_history_pool = []

            new_entity_path_pool = []
            new_relation_path_pool = []
            new_probs_pool = []
            predicted_entities = [batch_entity_actions[i][topk_actions[i]] for i in range(len(batch_entity_actions))]
            predicted_relations = [batch_relation_actions[i][topk_actions[i]] for i in range(len(batch_entity_actions))]
            for i in range(len(path_pool[1])):
                entity_path = path_pool[1][i]
                relation_path = path_pool[0][i]
                probs_path = probs_pool[i]
                predicted_instance_entities = predicted_entities[i]
                predicted_instance_relations = predicted_relations[i]
                instance_probs = topk_probs[i]
                for j in range(k):
                    new_entity_path_pool.append(entity_path+[predicted_instance_entities[j].item()])
                    new_relation_path_pool.append(relation_path+[predicted_instance_relations[j].item()])
                    new_probs_pool.append(probs_path+[instance_probs[j].item()])
                    dialogue_history_pool.append(dialogue_history[i])
            
            dialogue_history = torch.stack(dialogue_history_pool)
            probs_pool = new_probs_pool[:]
            path_pool = new_relation_path_pool[:], new_entity_path_pool[:]

        # new_probs_pool = [[-np.log(p+1e-60) for p in prob] for prob in new_probs_pool]
        rel_probs_pool = new_probs_pool[:]
        
        # new_probs_pool, new_entity_path_pool = fuse_edges(new_probs_pool, new_entity_path_pool)
        new_probs_pool = [reduce(lambda x, y: x*y, probs) for probs in new_probs_pool]
        probs_relation_entity = zip(new_probs_pool, new_entity_path_pool)
        probs_relation_entity = sorted(probs_relation_entity, key=lambda x:x[0], reverse=True)
        probs_relation_entity = zip(*probs_relation_entity)
        probs_relation_entity = [list(a) for a in probs_relation_entity]
        probs_pool , path_pool = probs_relation_entity[0], probs_relation_entity[1]
        true_path = true_entity_path[0]
        # true_path = (true_path[0], true_path[1], true_path[2])
        calculate_metrics(path_pool, probs_pool, true_path)
        relation_accuracy(new_relation_path_pool, rel_probs_pool, true_relation_path[0])


def predict_paths(policy_file, ConvKGDatasetLoaderTest, opt):
    print('Predicting paths...')
    pretrain_sd = torch.load(policy_file)
    model = ConvKGTrainer(opt).to(opt["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt["device"])

    K = [[2, 5, 5], [2, 5, 10], [2, 5, 25], [2, 5, 50], 
        [2, 10, 5], [2, 10, 10], [2, 10, 25], [2, 10, 50],
        [2, 25, 5], [2, 25, 10], [2, 25, 25], [2, 25, 50], 
        [2, 50, 5], [2, 50, 10], [2, 50, 25], [2, 50, 50], ]
    K = [[2, 25, 25]]
    # data_loader = iter(ConvKGDatasetLoaderTest)
    
    for ks in K:
        print(ks)
        data_loader = iter(ConvKGDatasetLoaderTest)
        with tqdm(total=len(data_loader)) as progress_bar:
            while(1):
                try:
                    batch=next(data_loader)
                except StopIteration:
                    break
                except:
                    continue
                batch_beam_search(model, batch, opt["device"], topk = ks, opt=opt)
                # if recall_entity["counts"]>1000:
                #     break
                progress_bar.update(1)
                
            for k, v in recall_entity.items():
                if "@" in k:
                    recall_entity[k] /= recall_entity["counts"]

            for k, v in recall_path.items():
                if "@" in k:
                    recall_path[k] /= recall_path["counts"]
            
            for k, v in recall_relation.items():
                if "@" in k:
                    recall_relation[k] /= recall_relation["counts"]
                

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_id', 
                        type=str, 
                        required=True,
                        help='Path to the training dataset text file')
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True,
                        help='Path to the training dataset text file')
    parser.add_argument('--data_directory', 
                        type=str, 
                        required=True,
                        help='Path to the training dataset text file')
    args = parser.parse_args()

    split_id = args.split_id
    model = args.model_name
    data_directory = args.data_directory

    data_directory = data_directory
    data_directory_init = data_directory
    splits_directory = "../../datasets/splits/split_"+split_id+"/"

    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialogue2dialogueId": data_directory+"dialogue2dialogueId.pkl", "dialogueId2AlbertRep": data_directory+"dialogueId2SBertRep.pkl",
                    "dataset": splits_directory+"dataset_test.pkl", "knowledge_graph": data_directory_init+"opendialkg_triples.txt", "device": device,
                    "max_dialogue_history": 3, "triple2tripleId": data_directory_init+"triple2tripleId.pkl", "dialogueId2dialogueRep_ranker": data_directory_init + "dialogue_embeddings.pkl",
                    "test": True}
    
    ConvKG_dataset_train = ConvKGDataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    self_loop_id = ConvKG_dataset_train.relation2relationId["self loop"]

    # triple_embeddings = load_pickle_file(data_directory_init + "/triples_embeddings.pkl")
    # triple_embeddings = triple_embeddings.to("cpu")
    opt_model = {"n_entity": len(ConvKG_dataset_train.entity2entityId)+1, "n_relation": len(ConvKG_dataset_train.relation2relationId)+1, "graph": ConvKG_dataset_train.graph,
                "entity2entityId": opt_dataset_train["entity2entityId"], "entity_embedding_path": opt_dataset_train["entity_embeddings"], "self_loop_id": self_loop_id,
                "entity_embeddings": ConvKG_dataset_train.entity_embeddings, "relation_embeddings": ConvKG_dataset_train.relation_embeddings, 
                "device": device, "entity_dim": 768, "relation_dim": 768, "dialogue_dim": 768, "model_directory": "models/", "model_name": "Pretrained_Reward_Agent",
                "batch_size":1, "epochs": 200, "max_path_length": 3, "state_dim": 60, "num_heads": 4, "entropy_weight": 0.1, "gamma": 0.99, "lr": 1e-4,
                "pretrained_actor_model": "Cloning_Model_actor_400", "clip":5, "alpha": 0.1, "test": True,
                "pretrained_discriminator_model": "Cloning_Model_discriminator_100", "pretrained_critic_model": "Cloning_Model_critic_20", "max_acts": 500}

    ConvKGDatasetLoaderTrain = DataLoader(ConvKG_dataset_train, batch_size=opt_model["batch_size"], shuffle=False, num_workers=0, collate_fn=ConvKG_collate)
    
    policy_file = model

    predict_paths(policy_file, ConvKGDatasetLoaderTrain, opt_model)
    