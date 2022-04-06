from collections import defaultdict
import dill as pickle
import dgl
import torch
from parlai.utils.misc import round_sigfigs

def load_pickle_file(location):
    with open(location, "rb") as f:
        pickle_variable = pickle.load(f)
    return pickle_variable

def _read_knowledge_graph_KBRD(kg_file, entity2entityId, relation2relationId, isDict = False):

    kg = load_pickle_file(kg_file)
    triples = _get_triples(kg)

    heads = []
    tails = []
    relations = []
    for line in triples:
        heads.append(line[0])
        tails.append(line[2])
        relations.append(line[1])
        
    return (heads, tails, relations)

def _read_knowledge_graph_dialkg(kg_file, entity2entityId, relation2relationId):
    triples = set()

    heads = []
    tails = []
    relations = []
    entities = []

    for line in open(kg_file, "r"):
        line = line[:-1].split("\t")
        h, r, t = line[0], line[1], line[2]
        entities.append(h)
        entities.append(t)
        heads.append(entity2entityId[h])
        tails.append(entity2entityId[t])
        relations.append(relation2relationId[r])
    entities = set(entities)
    for entity in entities:
        r = relation2relationId["self loop"]
        h = entity2entityId[entity]
        relations.append(r)
        heads.append(h)
        tails.append(h)
        
    return heads, tails, relations


def _get_triples(kg):
    triples = []
    for entity in kg:
        for relation, tail in kg[entity]:
            if entity != tail:
                triples.append([entity, relation, tail])
    return triples

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    entityIds = [v for k, v in entity2entityId.items()]
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = int(line[0])
            if entity not in entityIds:
                continue
            entityId = entity
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

def _edge_list(kg):
    edge_list = []
    for entity in kg.keys():
        for tail_and_relation in kg[entity]:
            if entity != tail_and_relation[1]:
                edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 10 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 10], len(relation_idx)

def _make_dgl_graph(heads, tails, relations):
    graph = dgl.graph((heads, tails))
    graph.edata["edge_type"] = torch.tensor(relations)
    graph.ndata["nodeId"] = graph.nodes()
    return graph

def _find_entity_path(paths):
    entity_path = []
    for path in paths:
        if not len(entity_path):
            entity_path += [path[0], path[-1]]
        else:
            entity_path += [path[-1]]

    if len(entity_path)==2:
        entity_path += [paths[0][-1]]
    return entity_path

def _find_relation_entity_path(paths, self_loop_id):
    relation_entity_path = []
    for path in paths:
        if not len(relation_entity_path):
            relation_entity_path.append((0, path[0]))
            relation_entity_path.append((path[1], path[2]))
        else:
            relation_entity_path.append((path[1], path[2]))
    if len(relation_entity_path)==2:
        relation_entity_path.append((self_loop_id, paths[0][-1]))

    return relation_entity_path

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId)+1, dim)
    entityIds = [v for k, v in entity2entityId.items()]
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            if "reverse" in line:
                continue
            line = line.split('\t')
            entity = int(line[0])
            if entity not in entityIds:
                continue
            entityId = entity
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

def reset_metrics(trainer):
    for key in trainer.metrics:
        trainer.metrics[key] = 0.0
    for key in trainer.counts:
        trainer.counts[key] = 0
    return trainer.metrics, trainer.counts

def report(trainer):
    m = {}
    # Top-k recommendation Recall
    for x in sorted(trainer.metrics):
        m[x] = trainer.metrics[x] / trainer.counts[x]

    for k, v in m.items():
        m[k] = round_sigfigs(v, 4)
    return m

