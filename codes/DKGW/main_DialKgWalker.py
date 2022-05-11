from __future__ import print_function, division
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
split_id = sys.argv[1]
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from dataset import DialKGDataset, ToTensor, dialkg_collate
from DialKGWalker_build import KGPathWalkerModel

import torch

from skimage import io, transform
import numpy as np
import pickle
import glob
from tqdm import tqdm

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import argparse



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def main(args):
    split_id = args.split_id
    data_directory = args.data_directory
    # data_directory = "../../datasets/dataset_dkgw/"
    splits_directory = "../../datasets/splits/split_"+split_id+"/"
    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "word2wordId": data_directory+"word2wordId.pkl", "wordId2wordEmb": data_directory+"wordId2wordEmb.pkl",
                    "dataset": splits_directory+"dataset_train.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 2, "n_max": 100, "max_dialogue_history": 3}
    opt_dataset_dev = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "word2wordId": data_directory+"word2wordId.pkl", "wordId2wordEmb": data_directory+"wordId2wordEmb.pkl",
                    "dataset": splits_directory+"dataset_valid.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 2, "n_max": 100, "max_dialogue_history": 3}

    # Dataset Preparation
    DialKG_dataset_train = DialKGDataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    DialKG_dataset_dev = DialKGDataset(opt=opt_dataset_dev, transform=transforms.Compose([ToTensor(opt_dataset_dev)]))

    # x = DialKG_dataset_train[100]
    opt_model = {"n_entity": len(DialKG_dataset_train.entity2entityId)+1, "n_relation": len(DialKG_dataset_train.relation2relationId)+1,
                "entity2entityId": opt_dataset_train["entity2entityId"], "entity_embedding_path": opt_dataset_train["entity_embeddings"],
                "entity_embeddings": DialKG_dataset_train.entity_embeddings, "relation_embeddings": DialKG_dataset_train.relation_embeddings, "word_embeddings": DialKG_dataset_train.word_embeddings,
                "hidden_dim":128, "word_dim": 300, "batch_size":10, "device": device, "lr": 1e-2, "lr_reduction_factor":0.1,
                "epochs": 25, "n_hops": 2, "model_directory": "models/", "model_name": "model_"+split_id}

    DialKGDatasetLoaderTrain = DataLoader(DialKG_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=dialkg_collate)
    DialKGDatasetLoaderDev = DataLoader(DialKG_dataset_dev, batch_size=8, shuffle=True, num_workers=0, collate_fn=dialkg_collate)

    walker = KGPathWalkerModel(opt_model)
    walker.to(device)

    walker.train_model(DialKGDatasetLoaderTrain, DialKGDatasetLoaderDev)


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
