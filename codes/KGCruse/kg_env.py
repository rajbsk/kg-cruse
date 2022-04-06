from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
import numpy as np
from time import time
from datetime import datetime

import dgl
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.transform import add_self_loop
from dgl.sampling import sample_neighbors
from torch.nn.utils.rnn import pad_sequence
cnt=0
class BatchKGEnvironment(object):
    def __init__(self, graph, opt):
        self.graph = graph
        self.n_entity = opt["n_entity"]
        self.n_relation = opt["n_relation"]
        self.max_path_length = opt["max_path_length"]
        self.device=opt["device"]
        self.max_acts = opt["max_acts"]
        self.self_loop_id = opt["self_loop_id"]

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self.batch_entity_history = None
        self.batch_dialogue_history = None
        self._done = False

    def _get_actions(self, source):
        neighbors_graph = dgl.sampling.sample_neighbors(self.graph, source.to("cpu"), -1, edge_dir = "out") #Samples a subgraph containing all the outgoing nodes of the source entity
        neighbor_relations, neighbor_entities = neighbors_graph.edata["edge_type"], neighbors_graph.edges()[1] #Gets the outgoing relations and nodes
        
        top_ents = torch.randperm(min(len(neighbor_entities), self.max_acts))[: self.max_acts]
        relation_candidate, entity_candidate = neighbor_relations[top_ents], neighbor_entities[top_ents] #get the top performing nodes and relations.

        return relation_candidate, entity_candidate
        
    def _batch_get_actions(self, batch_path, done):
        global cnt
        batch_entity_actions = []
        batch_relation_actions = []

        for i in range(len(batch_path[1])):
            source_entity = torch.tensor(batch_path[1][i][-1], dtype=torch.int64) #finds the last entity in the path already traversed
            # dialogue_representation = dialogue_representations[i] #get the scores of all the entities relevant to this dialogue context
            relation_candidate, entity_candidate = self._get_actions(source_entity)
            relations = relation_candidate.tolist()
            entities = entity_candidate.tolist()
            relations.append(self.self_loop_id)
            entities.append(batch_path[1][i][-1])
            relation_candidate = torch.tensor(relations, dtype=torch.int64)
            entity_candidate = torch.tensor(entities, dtype=torch.int64)

            batch_relation_actions.append(relation_candidate)
            batch_entity_actions.append(entity_candidate)

        return batch_relation_actions, batch_entity_actions

    def _batch_get_state(self):
        return self.batch_dialogue_history, self.batch_entity_history, self._batch_path

    def _get_reward(self, path, true_entity, time):
        res = 1.0*(path==true_entity)
        return res
        
    def _batch_get_reward(self, batch_path, batch_true_entity, time):
        predicted_batch_path_entities = [path[-1] for path in batch_path]
        batch_reward = [(self._get_reward(predicted_batch_path_entities[i], batch_true_entity[i], time)) for i in range(len(batch_path))]
        return np.array(batch_reward)

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or len(self._batch_path[0][0]) >= self.max_path_length

    def reset(self, batch_entity_history, batch_dialogue_history):
        
        self.batch_entity_history = batch_entity_history
        self.batch_dialogue_history = batch_dialogue_history
        self._batch_path = ([[] for _ in range(len(batch_entity_history))], [[] for _ in range(len(batch_entity_history))])
        self._done = False
        self._batch_curr_state = self._batch_get_state()

        curr_entity_actions = [torch.tensor(entity_history, dtype=torch.int64) for entity_history in batch_entity_history]
        curr_relation_actions = [torch.zeros(len(curr_entity_actions[i]), dtype=torch.int64) for i in range(len(curr_entity_actions))]
        
        self._batch_curr_actions = curr_relation_actions, curr_entity_actions
        self._batch_curr_reward = 0

        return self._batch_curr_state

    def batch_step(self, batch_act, batch_target_entity, time):
        """
        Args:
            batch_act: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act) == len(self._batch_path[1])

        # Execute batch actions.
        batch_curr_relation_action, batch_curr_entity_action = self._batch_curr_actions
        for i in range(len(batch_act)):
            act_idx = batch_act[i]
            # if act_idx >= len(batch_curr_relation_action):
            #     act_idx = 0
            act_relation = batch_curr_relation_action[i][act_idx]
            act_entity = batch_curr_entity_action[i][act_idx]
            self._batch_path[0][i].append(act_relation)
            self._batch_path[1][i].append(act_entity)

        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state()
        if time < self.max_path_length:
            self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        else:
            self._batch_curr_actions = None
        self._batch_curr_reward = self._batch_get_reward(self._batch_path[1], batch_target_entity, time)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self):
        """Return action masks of size [bs, actions]."""
        _, entity_candidates = self._batch_curr_actions
        entity_candidates = pad_sequence(entity_candidates, batch_first=True)
        batch_action_mask = entity_candidates != 0
        return batch_action_mask

