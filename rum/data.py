import numpy as np
import pandas as pd
from dgl.data import DGLDataset
from dgllife.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
        smiles_to_bigraph,mol_to_bigraph
    )
import torch
from .random_walk import node2vec_random_walk,uniform_random_walk_,uniform_random_walk
from dgl.dataloading import GraphDataLoader
import dgl
from rdkit import Chem
# from dgllife.utils import get_mol_3d_coordinates
# from rdkit.Chem import rdDetermineBonds,Get3DDistanceMatrix
from .utils import cleanup_qm9_xyz
from cugraph_dgl.convert import cugraph_storage_from_heterograph
HAR2EV = 27.2113825435      # 1 Hartree = 27.2114 eV 
KCALMOL2EV = 0.04336414     # 1 kcal/mol = 0.043363 eV

class qm9(DGLDataset):
    def __init__(self,dataset
                 ,args):
        super().__init__(name="qm9")
        self.path = '/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/data/qm9'
        # Load the npz file into memory as a dictionary
        with np.load(f'{self.path}/qm9_{dataset}.npz', allow_pickle=True) as data:
            self.df = {key: data[key] for key in data.files}
        self.label = args.label
        self.n_sample = args.num_samples
        self.length = args.length
        # self.mode = args.mode
    def __len__(self):
        return len(self.df['smiles'])
    def __getitem__(self, idx):
        smiles = self.df['smiles'][idx]
        ##TODO : Need better featurizer

        g = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer("h0"), edge_featurizer=CanonicalBondFeaturizer("e0"),explicit_hydrogens=True)
        y = np.array([self.df[l][idx] for l in self.label]).T
        ## Random walk
        # walks,eids = node2vec_random_walk(g, self.n_sample, self.length,mode=self.mode)
        # walks,eids = uniform_random_walk_(g, self.n_sample, self.length)
        walks, eids = None,None
        # g = cugraph_storage_from_heterograph(g)
        ## Convert 
        y = torch.tensor(y).float()
        return g, y,walks,eids
class qm9_xyz(DGLDataset):
    def __init__(self,dataset
                 ,args):
        super().__init__(name="qm9_xyz")
        self.path = f'/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/data/qm9'
        self.xyz_dir = f'/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/qm9_xyz'
        with np.load(f'{self.path}/qm9_{dataset}.npz',allow_pickle=True) as data:
            self.df = {key: data[key] for key in data.files}
        self.label = args.label
        self.n_sample = args.num_samples
        self.length = args.length
        self.mode = args.mode
        
    def __len__(self):
        return len(self.df['mol_id'])
    def __getitem__(self, idx):

        id = self.df['mol_id'][idx].split('_')[-1]
        id = id.zfill(6)
        ind,gdb_smi,relax_smi = cleanup_qm9_xyz(f'{self.xyz_dir}/dsgdb9nsd_{id}.xyz')
        raw_mol = Chem.MolFromXYZBlock(ind)
        conn_mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineConnectivity(conn_mol)
        # smiles = self.df['smiles'][idx]
        # g = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer("h0"), edge_featurizer=CanonicalBondFeaturizer("e0"))
        g = mol_to_bigraph(conn_mol, node_featurizer=CanonicalAtomFeaturizer('h0'), edge_featurizer=CanonicalBondFeaturizer('e0'))

        coords = get_mol_3d_coordinates(conn_mol)
        node_feats = g.ndata['h0']
        node_feats = torch.cat([node_feats,torch.tensor(coords).float()],dim=-1)
        g.ndata['h0'] = node_feats

        u,v = g.edges()
        distance_matrix= Get3DDistanceMatrix(conn_mol)
        edge_length = torch.tensor([distance_matrix[u[i],v[i]] for i in range(g.number_of_edges())])
        
        g.edata['el'] = edge_length

        y = np.array([self.df[l][idx] for l in self.label]).T
        y = torch.tensor(y).float()
        walks, eids = None,None

        ## Convert 
        return g, y, walks, eids
class qm9_graph(DGLDataset):
    def __init__(self,dataset,args):
        super().__init__(name="qm9_graph")
        self.path = f'/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/data/qm9'
        self.graph_dir = f'/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/qm9_graph'
        with np.load(f'{self.path}/qm9_{dataset}.npz',allow_pickle=True) as data:
            self.df = {key: data[key] for key in data.files}
        self.label = ['mu']
    def __len__(self):
        return len(self.df['mol_id'])
    def __getitem__(self, idx):
        id = self.df['mol_id'][idx].split('_')[-1]
        id = id.zfill(6)
        g = dgl.load_graphs(f'{self.graph_dir}/dsgdb9nsd_{id}.bin')[0][0]

        ##Remove atom position
        g.ndata['h0'] = g.ndata['h0'][:,:-3]
        y = np.array([self.df[l][idx] for l in self.label]).T
        y = torch.tensor(y).float()
        walks, eids = None,None
        return g, y, walks, eids
def collate(batch):
    g,y,walks,eids = zip(*batch)
    batch_g = dgl.batch(g)
    y = torch.vstack(y)

    # walks = torch.cat([walk for walk in walks],dim=1)
    # eids = torch.cat([eid for eid in eids],dim=1)
    walks,eids = None,None
    return batch_g,y,walks,eids
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="qm9")
    parser.add_argument("--label", type=str, default="gap")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--normalize_label", type=bool, default=True)
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mode", type=str, default='Random')
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--length", type=int, default=7)
    parser.add_argument("--optimizer", type=str, default="AdEMAMix")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--self_supervise_weight", type=float, default=0.01)
    parser.add_argument("--consistency_weight", type=float, default=0.01)
    parser.add_argument("--consistency_temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--warmup_ratio", type=float, default=2) ## 5 epochs warmup
    args = parser.parse_args()
    args.label = ['gap']
    data = qm9('small_train',args)
    g,y,walks,eids = data[0]
    print(g)
    print(type(walks))