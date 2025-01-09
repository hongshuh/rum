import numpy as np
import pandas as pd
from dgl.data import DGLDataset
from dgllife.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
        smiles_to_bigraph,
    )
import torch
from .random_walk import node2vec_random_walk,uniform_random_walk
from dgl.dataloading import GraphDataLoader
import dgl

HAR2EV = 27.2113825435      # 1 Hartree = 27.2114 eV 
KCALMOL2EV = 0.04336414     # 1 kcal/mol = 0.043363 eV

class qm9_(DGLDataset):
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
    def __len__(self):
        return len(self.df['smiles'])
    def __getitem__(self, idx):
        smiles = self.df['smiles'][idx]
        ##TODO : Need better featurizer
        g = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer("h0"), edge_featurizer=CanonicalBondFeaturizer("e0"))
        y = np.array([self.df[l][idx] for l in self.label]).T
        ## Random walk
        # walks,eids = node2vec_random_walk(g, self.n_sample, self.length,mode='Mix')
        # walks_dict = {'walks':walks,'eids':eids}
        ## Convert 
        y = torch.tensor(y).float()
        return g, y

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
        self.mode = args.mode
    def __len__(self):
        return len(self.df['smiles'])
    def __getitem__(self, idx):
        smiles = self.df['smiles'][idx]
        ##TODO : Need better featurizer
        g = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer("h0"), edge_featurizer=CanonicalBondFeaturizer("e0"))
        y = np.array([self.df[l][idx] for l in self.label]).T
        ## Random walk
        # walks,eids = node2vec_random_walk(g, self.n_sample, self.length,mode=self.mode)
        # walks,eids = uniform_random_walk(g, self.n_sample, self.length)
        walks, eids = None,None
        ## Convert 
        y = torch.tensor(y).float()
        return g, y,walks,eids
class qm9_xyz(DGLDataset):
    def __init__(self,dataset
                 ,label):
        super().__init__(name="qm9_xyz")
        self.path = f'/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/qm9_xyz'
        self.df = pd.read_csv(f'/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/data/qm9/qm9.csv')
        self.label = label
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        g = smiles_to_bigraph(data['smiles'], node_featurizer=CanonicalAtomFeaturizer("h0"), edge_featurizer=CanonicalBondFeaturizer("e0"))
        y = data[self.label].astype('float32')

        ## Convert 
        return g, y

def collate(batch):
    g,y,walks,eids = zip(*batch)
    batch_g = dgl.batch(g)
    y = torch.vstack(y)
    # walks = torch.cat([walk for walk in walks],dim=1)
    # eids = torch.cat([eid for eid in eids],dim=1)
    walks,eids = None,None
    return batch_g,y,walks,eids
if __name__ == "__main__":
    pass