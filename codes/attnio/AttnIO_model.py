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

class AttnIO(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, entity_embeddings, relation_embeddings, feat_drop=0., attn_drop=0., negative_slope=0.01, allow_zero_in_degree=False):
        super(AttnIO, self).__init__()

        self._num_heads = num_heads
        self._in_src_feats = in_feats
        self._in_dst_feats = out_feats
        self.in_feats = in_feats
        self._out_feats = out_feats
        self._head_out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # Embeddings
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        # Inflow Params
        self.fc = nn.Linear(self._in_src_feats, self._out_feats, bias=False)
        self.w_m = nn.Parameter(torch.FloatTensor(size=(self.in_feats, self._out_feats)))
        self.w_q = nn.Parameter(torch.FloatTensor(size=(num_heads, self._head_out_feats, self._head_out_feats)))
        self.w_k = nn.Parameter(torch.FloatTensor(size=(num_heads, self._head_out_feats, self._head_out_feats)))
        self.w_h_entity = nn.Parameter(torch.FloatTensor(size=(self._num_heads*self._out_feats, self._out_feats)))
        self.w_h_dialogue = nn.Parameter(torch.FloatTensor(size=(self.in_feats, self._out_feats)))

        # Outflow Params
        self.out_w_init = nn.Parameter(torch.FloatTensor(size=(self.in_feats, self._out_feats)))
        self.out_w_q = nn.Parameter(torch.FloatTensor(size=(num_heads, self._head_out_feats, self._head_out_feats)))
        self.out_w_k = nn.Parameter(torch.FloatTensor(size=(num_heads, self._head_out_feats, self._head_out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.w_m, gain=gain)
        nn.init.xavier_normal_(self.w_q, gain=gain)
        nn.init.xavier_normal_(self.w_k, gain=gain)
        nn.init.xavier_normal_(self.w_h_entity, gain=gain)
        nn.init.xavier_normal_(self.w_h_dialogue, gain=gain)
        nn.init.xavier_normal_(self.out_w_init, gain=gain)
        nn.init.xavier_normal_(self.out_w_q, gain=gain)
        nn.init.xavier_normal_(self.out_w_k, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def compute_inflow_features(self, graph, entity_features, relation_features, dialogue_context):
        feat_src = feat_dst = entity_features #Nodes X heads X features
        feat_dst = feat_dst.permute(1, 0, 2) #heads X nodes X features
        
        edst = torch.matmul(feat_dst, self.w_q) #(heads X Nodes X features) X (heads X features X features) -> (heads X Nodes X features)
        edst = edst.permute(1, 0, 2) # nodes X heads X features
        feat_dst = feat_dst.permute(1, 0, 2) # nodes X heads X features

        graph.srcdata.update({'ft_ent': feat_src, "in_el": feat_src})
        graph.dstdata.update({'in_er': edst})
        graph.edata.update({'ft_rel': relation_features})
        
        # Computing Attention Weights
        graph.apply_edges(fn.u_dot_v('in_el', 'in_er', 'in_e'))
        e = graph.edata.pop('in_e')
        graph.apply_edges(fn.e_dot_v('ft_rel', 'in_er', 'in_e'))
        re = graph.edata.pop('in_e')

        e = e + re
        e = self.leaky_relu(e)

        # compute edge softmax
        edge_attention = self.attn_drop(edge_softmax(graph, e, norm_by="dst"))
        graph.apply_edges(fn.u_add_e('ft_ent', 'ft_rel', "edge_message"))
        edge_message = graph.edata["edge_message"]*edge_attention
        graph.edata.update({"edge_message": edge_message})

        # message passing
        graph.update_all(fn.copy_edge("edge_message", "message"), fn.sum('message', 'ft_ent'))

        # rst contains the inflow nodes features
        rst = graph.ndata['ft_ent'].view(graph.num_nodes(), -1)
        entity_inflow_features = torch.mm(rst, self.w_h_entity) + torch.mm(dialogue_context, self.w_h_dialogue)
        return entity_inflow_features

    def compute_outflow_attention(self, graph, entity_features, relation_features):
        feat_src = feat_dst = entity_features
        feat_src = feat_src.permute(1, 0, 2)

        esrc = torch.matmul(feat_src, self.out_w_q)
        esrc = esrc.permute(1, 0, 2)
        feat_src = feat_src.permute(1, 0, 2)

        # Store the node features and attention features on the source and destination nodes
        graph.srcdata.update({"out_el": esrc})
        graph.dstdata.update({'ft': feat_src, 'out_er': feat_dst})
        graph.edata.update({'out_erel': relation_features})

        # compute edge attention with respect to the source node and the respective edge relation
        graph.apply_edges(fn.v_dot_u('out_er', 'out_el', 'out_e'))
        e = graph.edata.pop('out_e')
        graph.apply_edges(fn.e_dot_u('out_erel', 'out_er', 'out_e'))
        re = graph.edata.pop('out_e')

        e = e + re
        e = self.leaky_relu(e)

        # compute edge softmax
        edge_attention = (self.attn_drop(edge_softmax(graph, e, norm_by="src")))
        edge_attention = ((edge_attention.squeeze(-1)).sum(-1))/self._num_heads
        return edge_attention

    def forward(self, graph, seed_set, dialogue_context):
        rels = graph.edata["edge_type"]
        feat_rel = self.relation_embeddings(rels)
        feat = self.entity_embeddings(graph.ndata["nodeId"])
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = (self.fc(h_src)).unsqueeze(1)
        feat_rel = self.fc(feat_rel).unsqueeze(1)
        
        feat_src = feat_src.repeat(1, self._num_heads, 1)
        feat_rel = feat_rel.repeat(1, self._num_heads, 1)
        """
        INFLOW CODE: Code for the inflow section of AttnIO paper
        """
        for i in range(3):
            time_entity_features = self.compute_inflow_features(graph, feat_src, feat_rel, dialogue_context)
            graph.ndata.update({"entity_features_"+str(i): time_entity_features})
            feat_src = (time_entity_features.unsqueeze(1)).repeat(1, self._num_heads, 1)

        """
        OUTFLOW CODE: Code for the outflow section of AttnIO paper
        """
        # Adding Self-loops 1381 is the id of torch. self_loop
        # self_loop_vector = (torch.ones(graph.num_nodes(), dtype=torch.int64)*1381).to("cuda")
        # graph.add_edges(graph.nodes(), graph.nodes(), {"ft_rel": self.fc(self.relation_embeddings(self_loop_vector)).unsqueeze(1).repeat(1, self._num_heads, 1)})

        # Initial seed set attention weight calculation
        dialogue_context = torch.matmul(dialogue_context, self.out_w_init)
        conversation_seedset_attention = (torch.matmul(graph.ndata["entity_features_0"], dialogue_context.t())).squeeze(1)

        conversation_seedset_attention[seed_set] += 10000
        conversation_seedset_attention -= 10000
        # print(conversation_seedset_attention[seed_set])
        conversation_seedset_attention = softmax(conversation_seedset_attention)
        

        graph.ndata.update({"a_0": conversation_seedset_attention})

        # Outflow attention calculation
        for i in range(1, 3):
            entity_features = graph.ndata["entity_features_"+str(i)].unsqueeze(1).repeat(1, self._num_heads, 1)
            edge_attention = self.compute_outflow_attention(graph, entity_features, graph.edata["ft_rel"])
            graph.edata.update({"transition_probs_"+str(i): edge_attention})
            graph.update_all(fn.u_mul_e("a_"+str(i-1), "transition_probs_"+str(i), "time_"+str(i)), fn.sum("time_"+str(i), "a_"+str(i)))
            x = torch.sum(graph.ndata["a_"+str(i)])
            # graph.apply_edges(fn.u_mul_e("a_"+str(i-1), "time_"+str(i-1), "time_"+str(i)))
        return graph

# device = "cuda"
# node_features = Embedding(100926, 768)
# relation_features = Embedding(1382, 768)
# gat = AttnIO(in_feats=768, out_feats=80, num_heads=4, entity_embeddings=node_features, relation_embeddings=relation_features)
# gat = gat.to(device)

# g = dgl.graph((torch.empty(1000000, dtype=torch.long).random_(100926), torch.empty(1000000, dtype=torch.long).random_(100926)))
# relations = (torch.empty(1000000, dtype=torch.long).random_(1382))
# g.edata["edge_type"] = relations
# g.ndata["nodeId"] = g.nodes()
# g = g.to(device)

# dialogue_context = torch.rand(1, 768).to(device)
# seed_set = (torch.empty((4), dtype=torch.long).random_(100926)).to(device)
# start = time.time()
# res = gat(g, seed_set, dialogue_context)
