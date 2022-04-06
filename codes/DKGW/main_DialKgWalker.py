from __future__ import print_function, division
import sys
import os
split_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

from dataset import DialKGDataset, ToTensor, dialkg_collate
from DialKGWalker_build import KGPathWalkerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def main():
    data_directory = "../../datasets/dataset_baseline/"
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
    opt_dataset_test = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "word2wordId": data_directory+"word2wordId.pkl", "wordId2wordEmb": data_directory+"wordId2wordEmb.pkl",
                    "dataset": splits_directory+"dataset_test.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                    "n_hop": 2, "n_max": 100, "max_dialogue_history": 3}

    # Dataset Preparation
    DialKG_dataset_train = DialKGDataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    DialKG_dataset_dev = DialKGDataset(opt=opt_dataset_dev, transform=transforms.Compose([ToTensor(opt_dataset_dev)]))
    DialKG_dataset_test = DialKGDataset(opt=opt_dataset_test, transform=transforms.Compose([ToTensor(opt_dataset_test)]))

    # x = DialKG_dataset_train[100]
    opt_model = {"n_entity": len(DialKG_dataset_train.entity2entityId)+1, "n_relation": len(DialKG_dataset_train.relation2relationId)+1,
                "entity2entityId": opt_dataset_train["entity2entityId"], "entity_embedding_path": opt_dataset_train["entity_embeddings"],
                "entity_embeddings": DialKG_dataset_train.entity_embeddings, "relation_embeddings": DialKG_dataset_train.relation_embeddings, "word_embeddings": DialKG_dataset_train.word_embeddings,
                "hidden_dim":128, "word_dim": 300, "batch_size":10, "device": device, "lr": 1e-2, "lr_reduction_factor":0.1,
                "epochs": 25, "n_hops": 2, "model_directory": "models/", "model_name": "model_"+split_id}

    DialKGDatasetLoaderTrain = DataLoader(DialKG_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=dialkg_collate)
    DialKGDatasetLoaderDev = DataLoader(DialKG_dataset_dev, batch_size=8, shuffle=True, num_workers=0, collate_fn=dialkg_collate)
    DialKGDatasetLoaderTest = DataLoader(DialKG_dataset_test, batch_size=8, shuffle=True, num_workers=0, collate_fn=dialkg_collate)

    walker = KGPathWalkerModel(opt_model)
    walker.to(device)

    walker.train_model(DialKGDatasetLoaderTrain, DialKGDatasetLoaderDev)
    # AttnIO_model = AttnIOModel(opt_model)
    # AttnIO_model.to(device)
    
    # AttnIO_model.train_model(AttnIODatasetLoaderTrain, AttnIODatasetLoaderDev)

    # AttnIO_model.evaluate_model(AttnIODatasetLoaderTest)
    # AttnIO_model.load_state_dict(torch.load("models/AttnIO_l2_stable_scheduler_29"))

    # print(len(AttnIO_dataset_dev))
    # print(len(AttnIODatasetLoaderTest))

    # AttnIO_model.evaluate_model(AttnIODatasetLoaderDev)
    # AttnIO_model.evaluate_model(AttnIODatasetLoaderTest)

if __name__=="__main__":
    main()
