from __future__ import print_function, division
import sys
import os
split_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
from skimage import io, transform
import numpy as np
import pickle
import glob
from tqdm import tqdm
from functools import reduce

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import AttnIODataset, ToTensor, attnIO_collate
from AttnIO_build import AttnIOModel
from dgl.sampling import sample_neighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def get_actions(last_entity, graph):
    actions = sample_neighbors(g=graph, nodes = last_entity, fanout=-1, edge_dir="out")
    actions = actions.edges()[1]
    return actions

recall_entity = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
recall_path = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
recall_relation = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}

def calculate_metrics(paths, probs, true_path, edges):
    K = [1, 3, 5, 10, 25]
    x = edges
    filtered_paths = filter_paths(paths)
    
    entities = [path[-1] for path in filtered_paths]
    recall_entity["counts"] += 1
    recall_path["counts"] += 1

    x = 0
    for k in K:
        if true_path in paths[:k]:
            recall_path["counts@"+str(k)] += 1
        if true_path[-1] in entities[:k]:
            x = 1
            recall_entity["counts@"+str(k)] += 1
        if x==0 and k==25:
            p=10

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

    with torch.no_grad():
        model.eval()
        batch_dialogue_history, batch_entity_history, subgraph, true_entity_path = batch[0], batch[1], batch[2], batch[3]
        batch_dialogue_history = batch_dialogue_history[0].to(device)
        batch_entity_history = batch_entity_history[0].to(device)
        subgraph = subgraph[0].to(device)
        true_entity_path = true_entity_path[0].to(device)
        subgraph = model(batch_dialogue_history, batch_entity_history, subgraph)
        subgraph = subgraph.to("cpu")
        edges = subgraph.edges()

        done = False
        time = 0
        
        path_pool = []  # list of list, size=bs
        probs_pool = []
        for hop in range(3):
            if hop==0:
                k = min(topk[hop], len(batch_entity_history))
                probs = subgraph.ndata["a_0"].to("cpu")
                topk_probs, topk_actions = torch.topk(probs, k=k)
                for j in range(k):
                    path_pool.append([topk_actions[j].item()])
                    probs_pool.append([topk_probs[j].item()])
            else:
                new_path_pool = []
                new_probs_pool = []
                for i in range(len(path_pool)):
                    path = path_pool[i]
                    prob = probs_pool[i]
                    last_entity = path[-1]
                    neighbors = get_actions(last_entity, subgraph).tolist()
                    neighbor_scores = []
                    probs = subgraph.ndata["a_"+str(hop)].to("cpu")
                    x = torch.sum(probs)
                    for neighbor in neighbors:
                        neighbor_prob = probs[neighbor]
                        neighbor_scores.append(neighbor_prob.item())
                    neighbor_score = zip(neighbors, neighbor_scores)
                    neighbor_score = sorted(neighbor_score, key=lambda x:x[1], reverse=True)
                    neighbor_score = zip(*neighbor_score)
                    neighbor_score = [list(a) for a in neighbor_score]
                    neighbors, neighbor_scores = neighbor_score[0], neighbor_score[1]
                    k = min(topk[hop], len(set(neighbors)))

                    cnt = 0
                    for j in range(len(probs)):
                        node = neighbors[j]
                        prob_node = neighbor_scores[j]
                        if node in neighbors:
                            new_path_pool.append(path+[node])
                            new_probs_pool.append(prob+[prob_node])
                            cnt += 1
                            if cnt==k:
                                break
                path_pool = new_path_pool
                probs_pool = new_probs_pool

        # relation_accuracy(new_relation_path_pool, new_probs_pool, true_relation_path[0])
        # new_probs_pool = [[-np.log(p+1e-60) for p in prob] for prob in new_probs_pool]
        new_probs_pool = [reduce(lambda x, y: x*y, probs) for probs in new_probs_pool]
        probs_relation_entity = zip(new_probs_pool, path_pool)
        probs_relation_entity = sorted(probs_relation_entity, key=lambda x:x[0], reverse=True)
        probs_relation_entity = zip(*probs_relation_entity)
        probs_relation_entity = [list(a) for a in probs_relation_entity]
        probs_pool , path_pool = probs_relation_entity[0], probs_relation_entity[1]
        calculate_metrics(path_pool, probs_pool, true_entity_path.tolist(), edges)
        p=10

def predict_paths(policy_file, ConvKGDatasetLoaderTest, opt):
    print('Predicting paths...')
    pretrain_sd = torch.load(policy_file)
    model = AttnIOModel(opt).to(opt["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt["device"])

    K = [[2, 10, 1], [2, 10, 2], [2, 10, 5], [2, 10, 10], [2, 15, 1], [2, 15, 2], [2, 15, 5], [2, 15, 10],
        [2, 20, 1], [2, 20, 2], [2, 20, 5], [2, 20, 10], [2, 25, 1], [2, 25, 2], [2, 25, 5], [2, 25, 10], [2, 50, 50]]
    K = [[2, 15, 15]]
    i = 0
    with torch.no_grad():
        for ks in K:
            for batch in tqdm(ConvKGDatasetLoaderTest):
                # i+=1
                # if i==500:
                #     break
                batch_beam_search(model, batch, opt["device"], topk = ks, opt=opt)
            
            for k, v in recall_entity.items():
                if "@" in k:
                    recall_entity[k] /= recall_entity["counts"]

            for k, v in recall_path.items():
                if "@" in k:
                    recall_path[k] /= recall_path["counts"]
            
            # for k, v in recall_relation.items():
            #     if "@" in k:
            #         recall_relation[k] /= recall_relation["counts"]

            print(ks)
            path_res = str(recall_path["counts@1"]*100) + "\t" + str(recall_path["counts@3"]*100) + "\t" + str(recall_path["counts@5"]*100) + "\t" + str(recall_path["counts@10"]*100) + "\t" + str(recall_path["counts@25"]*100) + "\t" + str(recall_path["counts"])
            entity_res = str(recall_entity["counts@1"]*100) + "\t" + str(recall_entity["counts@3"]*100) + "\t" + str(recall_entity["counts@5"]*100) + "\t" + str(recall_entity["counts@10"]*100) + "\t" + str(recall_entity["counts@25"]*100) + "\t" + str(recall_path["counts"])
            print(path_res)
            print(entity_res)

            for k in recall_entity.keys():
                recall_entity[k] = 0
            for k in recall_path.keys():
                recall_path[k] = 0

if __name__ == '__main__':
    data_directory = "../../datasets/dataset_attnio/"
    splits_directory = "../../datasets/splits/split_"+split_id+"/"
    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialogue2dialogueId": data_directory+"dialogue2dialogueId.pkl", "dialogueId2AlbertRep": data_directory+"dialogueId2AlBertRep.pkl",
                    "dataset": splits_directory+"dataset_test.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 2, "n_max": 100, "max_dialogue_history": 3}
    
    AttnIO_dataset_train = AttnIODataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    opt_model = {"n_entity": len(AttnIO_dataset_train.entity2entityId)+1, "n_relation": len(AttnIO_dataset_train.relation2relationId)+1,
                "entity2entityId": opt_dataset_train["entity2entityId"], "entity_embedding_path": opt_dataset_train["entity_embeddings"],
                "entity_embeddings": AttnIO_dataset_train.entity_embeddings, "relation_embeddings": AttnIO_dataset_train.relation_embeddings,
                "out_dim":80, "in_dim": 768, "batch_size":1, "device": device, "lr": 5e-4, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,
                "epoch": 20, "model_directory": "models/", "model_name": "AttnIO", "clip": 5, "self_loop_id": AttnIO_dataset_train.relation2relationId["self loop"]}

    AttnIODatasetLoaderTrain = DataLoader(AttnIO_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=attnIO_collate)

    policy_file = opt_model["model_directory"] + "model_"+split_id+"_20"
    # path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    predict_paths(policy_file, AttnIODatasetLoaderTrain, opt_model)

