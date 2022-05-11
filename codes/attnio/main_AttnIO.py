from __future__ import print_function, division
import sys
import os
split_id = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from dataset import AttnIODataset, ToTensor, attnIO_collate
from AttnIO_build import AttnIOModel

import torch
from skimage import io, transform
import numpy as np
import pickle
import glob
from tqdm import tqdm
import argparse

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    data_directory = args.data_directory
    split_id = args.split_id
    # data_directory = "../../datasets/dataset_attnio/"
    splits_directory = "../../datasets/splits/split_"+split_id+"/"
    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialogue2dialogueId": data_directory+"dialogue2dialogueId.pkl", "dialogueId2AlbertRep": data_directory+"dialogueId2AlBertRep.pkl",
                    "dataset": splits_directory+"dataset_train.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 2, "n_max": 1000, "max_dialogue_history": 3}
    opt_dataset_dev = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialogue2dialogueId": data_directory+"dialogue2dialogueId.pkl", "dialogueId2AlbertRep": data_directory+"dialogueId2AlBertRep.pkl",
                    "dataset": splits_directory+"dataset_valid.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 2, "n_max": 1000, "max_dialogue_history": 3}

    # Dataset Preparation
    AttnIO_dataset_train = AttnIODataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    AttnIO_dataset_dev = AttnIODataset(opt=opt_dataset_dev, transform=transforms.Compose([ToTensor(opt_dataset_dev)]))

    opt_model = {"n_entity": len(AttnIO_dataset_train.entity2entityId)+1, "n_relation": len(AttnIO_dataset_train.relation2relationId)+1,
                "entity2entityId": opt_dataset_train["entity2entityId"], "entity_embedding_path": opt_dataset_train["entity_embeddings"],
                "entity_embeddings": AttnIO_dataset_train.entity_embeddings, "relation_embeddings": AttnIO_dataset_train.relation_embeddings,
                "out_dim":80, "in_dim": 768, "batch_size":8, "device": device, "lr": 5e-4, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,
                "epoch": 20, "model_directory": "models/", "model_name": "model_"+split_id, "clip": 5, "self_loop_id": AttnIO_dataset_train.relation2relationId["self loop"]}

    AttnIODatasetLoaderTrain = DataLoader(AttnIO_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=attnIO_collate)
    AttnIODatasetLoaderDev = DataLoader(AttnIO_dataset_dev, batch_size=8, shuffle=True, num_workers=0, collate_fn=attnIO_collate)

    AttnIO_model = AttnIOModel(opt_model)
    AttnIO_model.to(device)
    
    AttnIO_model.train_model(AttnIODatasetLoaderTrain, AttnIODatasetLoaderDev)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_id', 
                        type=str, 
                        required=True,
                        help='Path to the training dataset text file')
    parser.add_argument('--data_directory', 
                        type=str, 
                        required=True,
                        help='Dataset Directory')

    args = parser.parse_args()
    main(args)
