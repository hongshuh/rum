from re import sub
from typing import Callable
import torch
import dgl
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from .layers import RUMLayer, Consistency
from .utils import feat_nn,gate_nn

class WeightedAttentionWalk(torch.nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, feat_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)

       

        x = self.feat_nn(x)
        

        return x
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
        self.attn_feat = feat_nn(self.hidden_features, self.hidden_features, self.hidden_features)
        self.attn_gate = gate_nn(self.hidden_features, self.hidden_features)
        self.attn_pool = GlobalAttentionPooling(gate_nn=self.attn_gate,feat_nn=self.attn_feat)

    def forward(self, g, h, e=None, subsample=None):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # h = torch.nn.functional.tanh(h)
                h = torch.nn.SiLU()(h)
                h = h.mean(0)
            h, _loss = layer(g, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        # h = self.activation(h)
        # print(h.shape)

        ## pooling the random walks
        h = h.mean(0) ##TODO : Replace with attention pooling
        
        # print(h.shape)
        # g.ndata["h"] = h
        # h = dgl.sum_nodes(g, "h")
        ## pooling the graph
        # h = dgl.mean_nodes(g, "h") ##TODO : Replace with attention pooling

        # print(h.shape)
        h = self.attn_pool(g,h)
        ## output head
        h = self.fc_out(h)
        # print(h.shape)
        
        return h, loss