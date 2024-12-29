from re import sub
from typing import Callable
import torch
import dgl
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from .layers import RUMLayer, Consistency
from .utils import feat_nn,gate_nn

class Multi_Head_Attention(torch.nn.Module):
    def __init__(self,hidden,heads,gate,feat):
        super().__init__()
        assert hidden % heads == 0, "Input dimension must be divisible by the number of heads."
        self.heads = heads
        self.hidden = hidden
        self.gate = gate
        self.feat = feat
        self.attn_drop = torch.nn.Dropout(0.1)
    def forward(self,h):
        N,BN,D = h.shape # N: number of random walks, BN: batch size * number of nodes, D: hidden dimension
        f = self.feat(h)
        g = self.gate(h)
        f = f.view(N,BN,self.heads,D//self.heads)
        g = g.view(N,BN,self.heads,1)
        # Apply softmax over the gating scores for each head
        g_softmax = torch.nn.functional.softmax(g, dim=0)  # Shape: (N, BN, heads)
        g_softmax = self.attn_drop(g_softmax)
        # Compute weighted sum for each head
        h_heads = torch.sum(g_softmax * f,dim=0)  # Shape: (BN, heads, D//heads)
        # Concatenate the outputs of all heads
        h_out = h_heads.view(BN, self.hidden)  # Shape: (BN, D)
        return h_out
class AttentionWalk(torch.nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate, feat):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate = gate
        self.feat = feat

    def forward(self, h):
        print(h.shape)
        g = self.gate(h)
        print(g.shape)
        f = self.feat(h)
        g_softmax = torch.nn.functional.softmax(g, dim=0)
        h = torch.sum(g_softmax * f, dim=0)
        exit()
        return h
    def __repr__(self):
        return self.__class__.__name__
    
class RUMModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: Callable = torch.nn.ELU(),
            temperature=0.1,
            self_supervise_weight=0.05,
            consistency_weight=0.01,
            **kwargs,
    ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=True)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        for _ in range(depth):
            self.layers.append(RUMLayer(hidden_features, hidden_features, in_features, **kwargs))
        self.activation = activation
        self.consistency = Consistency(temperature=temperature)
        self.self_supervise_weight = self_supervise_weight
        self.consistency_weight = consistency_weight

    def forward(self, g, h, e=None, consistency_weight=None, subsample=None):
        g = g.local_var()
        if consistency_weight is None:
            consistency_weight = self.consistency_weight
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer(g, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        h = self.fc_out(h).softmax(-1)
        if self.training:
            _loss = self.consistency(h)
            _loss = _loss * consistency_weight
            loss = loss + _loss
        return h, loss

class RUMGraphRegressionModel(RUMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_out = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.hidden_features),
            self.activation,
            torch.nn.Linear(self.hidden_features, self.hidden_features),
            self.activation,
            torch.nn.Dropout(kwargs["dropout"]),
            torch.nn.Linear(self.hidden_features, self.out_features),
        )
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_features)
        self.heads = 8

        ## Graph Attention Pooling
        self.attn_feat = feat_nn(self.hidden_features, self.hidden_features, self.hidden_features)
        self.attn_gate = gate_nn(self.hidden_features, self.hidden_features, 1)
        self.attn_pool = GlobalAttentionPooling(gate_nn=self.attn_gate,feat_nn=self.attn_feat)

        ## Random Walk Attention
        # self.walk_gate = gate_nn(self.hidden_features, self.hidden_features, 1)
        # self.walk_feat = feat_nn(self.hidden_features, self.hidden_features, self.hidden_features)
        # self.walk_pool = AttentionWalk(gate=self.walk_gate,feat=self.walk_feat)

        ## Multi Head Attention
        self.walk_gate = gate_nn(self.hidden_features, self.hidden_features, self.heads)
        self.walk_feat = feat_nn(self.hidden_features, self.hidden_features, self.hidden_features)
        self.walk_pool = Multi_Head_Attention(self.hidden_features,self.heads,gate=self.walk_gate,feat=self.walk_feat)
    def forward(self, g, h, e=None, subsample=None):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # h = torch.nn.functional.tanh(h)
                h = torch.nn.SiLU()(h)
                ##TODO might want to change into attention aggregation and add skip connections
                h = h.mean(0)
            h, _loss = layer(g, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        # h = self.activation(h)
        # print(h.shape)
        h = self.LayerNorm(h)
        ## pooling the random walks
        #h = h.mean(0) # mean pooling
        h = self.walk_pool(h) # attention pooling

        ## pooling node into garph
        # g.ndata["h"] = h
        # h = dgl.sum_nodes(g, "h") # sum pooling
        # h = dgl.mean_nodes(g, "h") # mean pooling
        h = self.attn_pool(g,h) # attention pooling

        ## output head
        h = self.fc_out(h)
        return h, loss