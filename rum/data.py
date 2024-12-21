import pandas as pd
from dgl.data import DGLDataset
from dgllife.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
        smiles_to_bigraph,
    )
import torch
from dgl.dataloading import GraphDataLoader
HAR2EV = 27.2113825435      # 1 Hartree = 27.2114 eV 
KCALMOL2EV = 0.04336414     # 1 kcal/mol = 0.043363 eV

class qm9(DGLDataset):
    def __init__(self,dataset
                 ,label):
        super().__init__(name="qm9")
        self.path = '/nfs/turbo/coe-venkvis/hongshuh/rum/dataset/data/qm9'
        self.df = pd.read_csv(f'{self.path}/qm9_{dataset}.csv')
        self.label = label
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        g = smiles_to_bigraph(data['smiles'], node_featurizer=CanonicalAtomFeaturizer("h0"), edge_featurizer=CanonicalBondFeaturizer("e0"))
        y = data[self.label].astype('float32')

        ## Convert 
        return g, y

if __name__ == "__main__":
    from utils import Normalizer
    train_set = qm9('train','gap')
    train_loader = GraphDataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    # for g, y in train_loader:
    #     print(g)
    #     print(y)
    #     print(y.dtype)
    #     break
    train_label = torch.tensor(train_set.df[train_set.label].values,dtype=torch.float32,requires_grad=False)
    normalizer = Normalizer(train_label)
    print(normalizer.mean, normalizer.std,train_label.shape)
    
        