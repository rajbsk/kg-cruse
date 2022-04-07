import time
import pickle

import torch
from torch import nn
from torch.nn.functional import softmax, relu
from torch.nn import Embedding

import dgl
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.transform import add_self_loop
from dgl.sampling import sample_neighbors

def nodes_sum(nodes, prev, current):
    return {current: torch.sum(nodes.mailbox[prev], dim=1)}

class Inflow(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dial_size):
        super(Inflow, self).__init__()

        self._num_heads = num_heads
        self._in_src_feats = in_feats
        self._in_dst_feats = out_feats
        self.in_feats = in_feats
        self._out_feats = out_feats
        self._head_out_feats = out_feats
        self.dial_size = dial_size

        # Inflow Params
        self.w_m = nn.Parameter(torch.FloatTensor(size=(num_heads, self.in_feats, self._out_feats)))
        self.w_q = nn.Parameter(torch.FloatTensor(size=(num_heads, self.in_feats, self._head_out_feats)))
        self.w_k = nn.Parameter(torch.FloatTensor(size=(num_heads, self.in_feats, self._head_out_feats)))
        self.w_h_entity = nn.Parameter(torch.FloatTensor(size=(self._num_heads*self._out_feats, self._out_feats)))
        self.w_h_dialogue = nn.Parameter(torch.FloatTensor(size=(self.dial_size, self._out_feats)))

        self.leaky_relu = nn.LeakyReLU()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_m, gain=gain)
        nn.init.xavier_normal_(self.w_q, gain=gain)
        nn.init.xavier_normal_(self.w_k, gain=gain)
        nn.init.xavier_normal_(self.w_h_entity, gain=gain)
        nn.init.xavier_normal_(self.w_h_dialogue, gain=gain)
    
    def forward(self, graph, entity_features, relation_features, dialogue_context):
        feat_src = entity_features.repeat(1, self._num_heads, 1)
        feat_rel = relation_features.repeat(1, self._num_heads, 1)

        feat_dst = feat_src #Nodes X heads X features
        feat_src = feat_src.permute(1, 0, 2) #heads X nodes X features
        feat_dst = feat_dst.permute(1, 0, 2) #heads X nodes X features
        feat_rel = feat_rel.permute(1, 0, 2) #heads X relations X features

        # Equation 3b:
        feat_dest_attn = torch.matmul(feat_dst, self.w_q) #(heads X Nodes X features) X (heads X features X features) -> (heads X Nodes X features)
        feat_src_attn = torch.matmul(feat_src, self.w_k) #(heads X Nodes X features) X (heads X features X features) -> (heads X Nodes X features)
        feat_rel_attn = torch.matmul(feat_rel, self.w_k) #(heads X relations X features) X (heads X features X features) -> (heads X relations X features)

        # Equation 1:
        feat_src = torch.matmul(feat_src, self.w_m) #(heads X nodes X features) X (heads X features X features) -> (heads X nodes X features)
        feat_rel = torch.matmul(feat_rel, self.w_m) #(heads X relations X features) X (heads X features X features) -> (heads X relations X features)

        feat_dest_attn = feat_dest_attn.permute(1, 0, 2)
        feat_src_attn = feat_src_attn.permute(1, 0, 2)
        feat_rel_attn = feat_rel_attn.permute(1, 0, 2)
        feat_src = feat_src.permute(1, 0, 2)
        feat_rel = feat_rel.permute(1, 0, 2)
        
        graph.srcdata.update({'ft_ent': feat_src, "in_el": feat_src_attn})
        graph.dstdata.update({'in_er': feat_dest_attn})
        graph.edata.update({'ft_rel': feat_rel, 'in_rel': feat_rel_attn})
        
        # Computing Attention Weights. Equation 3a
        graph.apply_edges(fn.u_dot_v('in_el', 'in_er', 'in_e'))
        e = graph.edata.pop('in_e')
        graph.apply_edges(fn.e_dot_v('ft_rel', 'in_er', 'in_e'))
        re = graph.edata.pop('in_e')

        e = e + re
        e = self.leaky_relu(e)

        # compute edge softmax
        edge_attention = edge_softmax(graph, e, norm_by="dst")
        graph.apply_edges(fn.u_add_e('ft_ent', 'ft_rel', "edge_message"))
        edge_message = graph.edata["edge_message"]*edge_attention
        graph.edata.update({"edge_message": edge_message})

        # message passing. Equation 2
        graph.update_all(fn.copy_edge("edge_message", "message"), fn.sum('message', 'ft_ent'))

        # rst contains the inflow nodes features
        rst = graph.ndata['ft_ent'].view(graph.num_nodes(), -1)

        #Equation 5
        entity_inflow_features = torch.mm(rst, self.w_h_entity) + torch.mm(dialogue_context, self.w_h_dialogue)
        return entity_inflow_features

class Outflow(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(Outflow, self).__init__()

        self._num_heads = num_heads
        self._in_src_feats = in_feats
        self._in_dst_feats = out_feats
        self.in_feats = in_feats
        self._out_feats = out_feats
        self._head_out_feats = out_feats

        # Outflow Params
        self.w_q = nn.Parameter(torch.FloatTensor(size=(num_heads, self.in_feats, self._head_out_feats)))
        self.w_k = nn.Parameter(torch.FloatTensor(size=(num_heads, self.in_feats, self._head_out_feats)))

        self.leaky_relu = nn.LeakyReLU()
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_q, gain=gain)
        nn.init.xavier_normal_(self.w_k, gain=gain)
    
    def forward(self, graph, entity_features, relation_features):
        
        feat_src = entity_features.repeat(1, self._num_heads, 1)
        feat_rel = relation_features.repeat(1, self._num_heads, 1)

        feat_dst = feat_src
        feat_src = feat_src.permute(1, 0, 2)
        feat_dst = feat_dst.permute(1, 0, 2)
        feat_rel = feat_rel.permute(1, 0, 2)

        feat_dst_attn = torch.matmul(feat_dst, self.w_k)
        feat_rel_attn = torch.matmul(feat_rel, self.w_k)
        feat_src_attn = torch.matmul(feat_src, self.w_q)

        feat_dst_attn = feat_dst_attn.permute(1, 0, 2)
        feat_rel_attn = feat_rel_attn.permute(1, 0, 2)
        feat_src_attn = feat_src_attn.permute(1, 0, 2)
        feat_src = feat_src.permute(1, 0, 2)

        # Store the node features and attention features on the source and destination nodes
        graph.srcdata.update({"out_el": feat_src_attn})
        graph.dstdata.update({'ft': feat_src, 'out_er': feat_dst_attn})
        graph.edata.update({'out_erel': feat_rel_attn})

        # compute edge attention with respect to the source node and the respective edge relation
        graph.apply_edges(fn.v_dot_u('out_er', 'out_el', 'out_e'))
        e = graph.edata.pop('out_e')
        graph.apply_edges(fn.e_dot_u('out_erel', 'out_er', 'out_e'))
        re = graph.edata.pop('out_e')

        e = e + re
        e = self.leaky_relu(e)

        # compute edge softmax
        edge_attention = edge_softmax(graph, e, norm_by="src")
        edge_attention = edge_attention.squeeze(-1)
        edge_attention = edge_attention.sum(-1)
        edge_attention = ((edge_attention))/self._num_heads
        return edge_attention

class AttnIO(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, entity_embeddings, relation_embeddings, self_loop_id, device):
        super(AttnIO, self).__init__()

        self._num_heads = num_heads
        self.in_feats = in_feats
        self._out_feats = out_feats
        self.self_loop_id = self_loop_id
        self.device = device

        # Embeddings
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.fce = nn.Linear(self.in_feats, self._out_feats, bias=False)
        self.fcr = nn.Linear(self.in_feats, self._out_feats, bias=False)
        self.out_w_init = nn.Parameter(torch.FloatTensor(size=(self.in_feats, self._out_feats)))

        # Inflow Layers
        self.inflow_layer_1 = Inflow(self._out_feats, self._out_feats, self._num_heads, self.in_feats)
        self.inflow_layer_2 = Inflow(self._out_feats, self._out_feats, self._num_heads, self.in_feats)

        # Outflow Layers
        self.outflow_layer_1 = Outflow(self._out_feats, self._out_feats, self._num_heads)
        self.outflow_layer_2 = Outflow(self._out_feats, self._out_feats, self._num_heads)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.out_w_init, gain=gain)

    def forward(self, graph, seed_set, dialogue_context):
        rels = graph.edata["edge_type"]
        feat_rels = self.fcr(self.relation_embeddings.weight)
        feat = self.entity_embeddings(graph.ndata["nodeId"])
        feat_rel = feat_rels[rels]
        feat = self.fce(feat)
        
        context = torch.matmul(dialogue_context, self.out_w_init)
        conversation_seedset_attention = (torch.matmul(feat, context.t())).squeeze(1)

        conversation_seedset_attention[seed_set] += 10000
        conversation_seedset_attention -= 10000
        conversation_seedset_attention = softmax(conversation_seedset_attention)
    
        graph.ndata.update({"a_0": conversation_seedset_attention})

        feat_rel = feat_rel.unsqueeze(1)
        feat = feat.unsqueeze(1)
        inflow_t_1 = self.inflow_layer_1(graph, feat, feat_rel, dialogue_context)
        inflow_t_1 = inflow_t_1.unsqueeze(1)
        
        graph.add_edges(graph.nodes(), graph.nodes(), data={"edge_type": torch.ones(graph.num_nodes(), dtype=torch.int64).to(self.device)*(self.self_loop_id)})
        rels = graph.edata["edge_type"]
        feat_rel = feat_rels[rels].unsqueeze(1)
        outflow_t_1 = self.outflow_layer_1(graph, inflow_t_1, feat_rel)

        graph.edata.update({"transition_probs_1": outflow_t_1})
        graph.update_all(fn.u_mul_e("a_0", "transition_probs_1", "time_1"), fn.sum("time_1", "a_1"))
        # a_0 = graph.ndata["a_1"]

        graph = dgl.remove_self_loop(graph)
        rels = graph.edata["edge_type"]
        feat_rel = feat_rels[rels].unsqueeze(1)
        inflow_t_2 = self.inflow_layer_2(graph, feat, feat_rel, dialogue_context)
        inflow_t_2 = inflow_t_2.unsqueeze(1)

        graph.add_edges(graph.nodes(), graph.nodes(), data={"edge_type": torch.ones(graph.num_nodes(), dtype=torch.int64).to(self.device)*(self.self_loop_id)})
        rels = graph.edata["edge_type"]
        feat_rel = feat_rels[rels].unsqueeze(1)
        outflow_t_2 = self.outflow_layer_2(graph, inflow_t_2, feat_rel)

        graph.edata.update({"transition_probs_2": outflow_t_2})
        graph.update_all(fn.u_mul_e("a_1", "transition_probs_2", "time_2"), fn.sum("time_2", "a_2"))

        return graph

