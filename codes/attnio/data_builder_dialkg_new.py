import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from collections import defaultdict
import pandas as pd
import json
from ast import literal_eval
import re
import unicodedata
import dill as pickle
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from transformers import AlbertTokenizer, AlbertModel
from sentence_transformers import SentenceTransformer

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
encoder = AlbertModel.from_pretrained('albert-base-v2').to("cuda")
sentence_encoder = SentenceTransformer('bert-base-nli-mean-tokens')

def get_albert_representations(sentence):
    sentence_tokenized = tokenizer(sentence, return_tensors="pt").to("cuda")
    sentence_encoding = encoder(**sentence_tokenized)[0].detach().to("cpu")
    sentence_encoding = sentence_encoding[0][0]
    return sentence_encoding

def get_sentence_representations(sentence):
    sentences = [sentence]
    sentence_embedding = sentence_encoder.encode(sentences)[0]
    return torch.tensor(sentence_embedding)


def load_pickle_file(location):
    with open(location, "rb") as f:
        pickle_variable = pickle.load(f)
    return pickle_variable

def normalizeString(s):
    s = unicodeToAscii(s.strip())
    if s.startswith("The "):
        s = s[4:]
    if s.startswith("A "):
        s = s[2:]
        
#     s = re.sub(r"[.!]", r" ", s)
    s = re.sub("\(.+\)", " ", s)
    s = re.sub(r"[^a-zA-Z.!?]+\'", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def _extract_subkg(kg, seed_set, n_hop):
    subkg = defaultdict(list)
    subkg_hrt = set()

    ripple_set = []
    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []
        
        if h == 0:
            tails_of_last_hop = seed_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                h, r, t = entity, tail_and_relation[0], tail_and_relation[1]
                if (h, r, t) not in subkg_hrt:
                    subkg[h].append((r, t))
                    subkg_hrt.add((h, r, t))
                memories_h.append(h)
                memories_r.append(r)
                memories_t.append(t)

        ripple_set.append((memories_h, memories_r, memories_t))

    return list(subkg_hrt)

def process_opendialkg_dataset(messages):
    dataset = []
    utterances = []
    for dialogue in tqdm(messages):
        dialogue = literal_eval(dialogue)
        dialogue_history = []
        starting_entities = set()
        previous_sentence = ""

        for ti, turn in enumerate(dialogue):
            if 'action_id' in turn and turn['action_id'] == 'kgwalk/choose_path':
                kg_path = turn['metadata']['path'][1]

                if len(starting_entities) == 0:
                    starting_entities.add(kg_path[0][0])
                
                if kg_path[0][0] not in starting_entities:
                    raise KeyError('%s not found.' % kg_path[0][0])

                sample = [previous_sentence, dialogue_history[:], starting_entities, kg_path]
                if ti != 0:  # there are few samples where assistant chooses path from scratch, I discarded these turns.
                    dataset.append(sample)
                
                starting_entities = set()
                for triple in kg_path:
                    starting_entities.add(triple[2])
                
            elif "message" in turn:
                if len(previous_sentence) != 0:
                    dialogue_history.append(previous_sentence)
                previous_sentence = turn["message"]
                utterances.append(previous_sentence)

    return [dataset, utterances]

def read_opendialkg_dataset(dialogue_file, destination_folder):
    opendialkg_dataframe = pd.read_csv(dialogue_file)
    utterance2utteranceId = defaultdict(int)
    messages = opendialkg_dataframe["Messages"].tolist()

    train_messages, dev_test_messages = train_test_split(messages, test_size=0.3)
    dev_messages, test_messages = train_test_split(dev_test_messages, test_size=0.5)

    print(len(train_messages))
    print(len(dev_messages))
    print(len(test_messages))
    utterances = []
    dataset = []

    dataset_train, utterances_train = process_opendialkg_dataset(train_messages)
    dataset_dev, utterances_dev = process_opendialkg_dataset(dev_messages)
    dataset_test, utterances_test = process_opendialkg_dataset(test_messages)

    utterances = utterances_train + utterances_dev + utterances_test
    dataset = dataset_train + dataset_dev + dataset_test

    with open(destination_folder+"dataset_train.pkl", "wb") as f:
        pickle.dump(dataset_train, f)
    with open(destination_folder+"dataset_test.pkl", "wb") as f:
        pickle.dump(dataset_test, f)
    with open(destination_folder+"dataset_valid.pkl", "wb") as f:
        pickle.dump(dataset_dev, f)

    dialogue_histories = []
    for d in dataset:
        current_utterance = d[0]
        dialogue_history = d[1]
        dialogue_context = " ".join(dialogue_history[-2:] + [current_utterance])
        dialogue_histories.append(dialogue_context)

    idx = 1
    dialogue2dialogueId = defaultdict(int)
    for d in dialogue_histories:
        dialogue2dialogueId[d] = idx
        idx += 1
        
    idx = 1
    for utterance in set(utterances):
        if not utterance2utteranceId[utterance]:
            utterance2utteranceId[utterance] = idx
            idx += 1
    dialogueId2AlbertRep = defaultdict(lambda: torch.tensor)
    for dialogue, dialogueId in tqdm(dialogue2dialogueId.items()):
        dialogueId2AlbertRep[dialogueId] = get_albert_representations(dialogue)
    
    utteranceId2utterance = {v:k for k, v in utterance2utteranceId.items()}

    with open(destination_folder+"utterance2utteranceId.pkl", "wb") as f:
        pickle.dump(utterance2utteranceId, f)
    with open(destination_folder+"utteranceId2utterance.pkl", "wb") as f:
        pickle.dump(utteranceId2utterance, f)
    with open(destination_folder+"dialogue2dialogueId.pkl", "wb") as f:
        pickle.dump(dialogue2dialogueId, f)
    with open(destination_folder+"dialogueId2SBertRep.pkl", "wb") as f:
        pickle.dump(dialogueId2AlbertRep, f)


def encode_knowledge_graph(knowledge_graph, destination_folder):
    entity2entityId = defaultdict(int)
    relation2relationId = defaultdict(int)

    entityIdx = 1
    relationIdx = 1
    kg = defaultdict(list)
    entity_embeddings = []
    relation_embeddings = []

    for line in (open(knowledge_graph, "r")):
        head, relation, tail = line[:-1].split("\t")
        
        if not entity2entityId[head]:
            entity2entityId[head] = entityIdx
            entityIdx += 1

        if not entity2entityId[tail]:
            entity2entityId[tail] = entityIdx
            entityIdx += 1

        if not relation2relationId[relation]:
            relation2relationId[relation] = relationIdx
            relationIdx += 1
        kg[entity2entityId[head]].append([relation2relationId[relation], entity2entityId[tail]])

    relation2relationId["self loop"] = relationIdx
    entityId2entity = {v:k for k, v in entity2entityId.items()}
    relationId2relation = {v:k for k, v in relation2relationId.items()}

    for entityId in tqdm(sorted(entityId2entity.keys())):
        entity_embeddings.append(get_albert_representations(entityId2entity[entityId]))

    for relationId in tqdm(sorted(relationId2relation)):
        relation = re.sub("\~", "reverse ", relationId2relation[relationId])
        relation_embedding = get_albert_representations(relation)
        relation_embeddings.append(relation_embedding)

    relation_embeddings = [torch.zeros(768, dtype=torch.float32)] + relation_embeddings
    entity_embeddings = [torch.zeros(768, dtype=torch.float32)] + entity_embeddings

    relation_embeddings = torch.stack(relation_embeddings)
    entity_embeddings = torch.stack(entity_embeddings)


    with open(destination_folder+"entity2entityId.pkl", "wb") as f:
        pickle.dump(entity2entityId, f)
    
    with open(destination_folder+"relation2relationId.pkl", "wb") as f:
        pickle.dump(relation2relationId, f)
    
    with open(destination_folder+"entityId2entity.pkl", "wb") as f:
        pickle.dump(entityId2entity, f)
    
    with open(destination_folder+"relationId2relation.pkl", "wb") as f:
        pickle.dump(relationId2relation, f)
    
    with open(destination_folder+"entity_embeddings.pkl", "wb") as f:
        pickle.dump(entity_embeddings, f)
    
    with open(destination_folder+"relation_embeddings.pkl", "wb") as f:
        pickle.dump(relation_embeddings, f)

    return kg, entity2entityId, relation2relationId
        

def main(diaogue_history):
    
    knowledge_graph_file = "/home/ubuntu/dataset/opendialkg_triples.txt"
    dialogue_file = "/home/ubuntu/dataset/opendialkg.csv"
    destination_folder = "/home/ubuntu/dataset/"

    kg, entity2entityId, relation2relationId = encode_knowledge_graph(knowledge_graph=knowledge_graph_file, destination_folder = destination_folder)
    read_opendialkg_dataset(dialogue_file = dialogue_file, destination_folder=destination_folder)

if __name__=="__main__":
    # parser = argparse.ArgumentParser(description="Reading files for Knowledge Selection Project")
    # parser.add_argument("dialogue_history", help="Length of Dialogue History to be considered.")
    
    # args = parser.parse_args()
    # print(int(args.dialogue_history))
    main(3)

    # main()

