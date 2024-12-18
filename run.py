import numpy as np
import torch
import dgl
import math
from transformers import get_cosine_schedule_with_warmup
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)

def get_graphs(data):
    from dgllife.data import (
        ESOL,
        FreeSolv,
        Lipophilicity,
    )
    from dgllife.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
    )
    data = locals()[data](
        node_featurizer=CanonicalAtomFeaturizer("h0"),
        edge_featurizer=CanonicalBondFeaturizer("e0"),
    )
    from dgllife.utils import RandomSplitter
    splitter = RandomSplitter()
    data_train, data_valid, data_test = splitter.train_val_test_split(
        data, frac_train=0.8, frac_val=0.1, frac_test=0.1, 
        # random_state=args.seed,
    )
    return data_train, data_valid, data_test


def run(args):
    train_set, valid_set, test_set = get_graphs(args.data)
    data_train = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    data_valid = dgl.dataloading.GraphDataLoader(
        valid_set, batch_size=args.batch_size,
    )

    data_test = dgl.dataloading.GraphDataLoader(
        test_set, batch_size=args.batch_size,
    )

    _, g, y = next(iter(data_train))
    
    ## Wandb Init
    import wandb
    wandb.init(project=f"RUM_{args.data}", config=args)

    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g.ndata["h0"].shape[-1],
        out_features=1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        temperature=args.consistency_temperature,
        dropout=args.dropout,
        num_layers=1,
        self_supervise_weight=args.self_supervise_weight,
        consistency_weight=args.consistency_weight,
        activation=getattr(torch.nn, args.activation)(),
        edge_features=g.edata["e0"].shape[-1],
    )

    if torch.cuda.is_available():
        model = model.cuda()

    
    if args.optimizer == "AdEMAMix":
        from rum.ademamix import AdEMAMix
        optimizer = AdEMAMix(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = getattr(
            torch.optim,
            args.optimizer,
        )(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # scheduler
    steps_per_epoch = math.ceil(len(data_train) // args.batch_size)
    training_steps = steps_per_epoch * args.n_epochs
    warmup_steps = int(training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
    
    from rum.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Dummy validation
    mse_vl, mse_te = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for _, g, y in data_valid:
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            h_vl, _ = model(g, g.ndata["h0"], e=g.edata["e0"])
            mse_vl += torch.nn.functional.mse_loss(h_vl, y,reduction='sum').item()
        rmse_vl = np.sqrt(mse_vl/len(valid_set))
        for _, g, y in data_test:
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            h_te, _ = model(g, g.ndata["h0"], e=g.edata["e0"])
            mse_te += torch.nn.functional.mse_loss(h_te, y,reduction='sum').item()
        rmse_te = np.sqrt(mse_te/len(test_set)).item()
            
    rmse_vl_min, rmse_te_min = rmse_vl, rmse_te
    print(f'Dummy RMSE on validation and test set : {rmse_vl_min}, {rmse_te_min}')
    
    for idx in range(args.n_epochs):
        for _, g, y in data_train:
            model.train()
            if torch.cuda.is_available():
                g = g.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["h0"], e=g.edata["e0"])
            mse = torch.nn.functional.mse_loss(h, y)
            loss = loss + mse
            loss.backward()
            
            # logging loss to wandb
            wandb.log({"train loss": loss.item()})
            wandb.log({"train RMSE": torch.sqrt(mse).item()})

            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            mse_vl = 0.0
            for _, g, y in data_valid:
                if torch.cuda.is_available():
                    g = g.to("cuda")
                    y = y.to("cuda")
                h_vl, _ = model(g, g.ndata["h0"], e=g.edata["e0"])
                mse_vl += torch.nn.functional.mse_loss(h_vl, y,reduction='sum').item()
            rmse_vl = np.sqrt(mse_vl/len(valid_set))
            wandb.log({"RMSE_validation": rmse_vl})
            if early_stopping([rmse_vl]) and idx > 200:
                # print("Early stopping at epoch :", idx)
                break

            mse_te = 0.0
            for _, g, y in data_test:
                if torch.cuda.is_available():
                    g = g.to("cuda")
                    y = y.to("cuda")
                h_te, _ = model(g, g.ndata["h0"], e=g.edata["e0"])
                mse_te += torch.nn.functional.mse_loss(h_te, y,reduction='sum').item()
            ## Average the RMSE
            rmse_te =np.sqrt(mse_te/len(test_set))
            wandb.log({"RMSE_test": rmse_te})

            # print(rmse_vl, rmse_te)
            if rmse_vl < rmse_vl_min:
                rmse_vl_min = rmse_vl
                rmse_te_min = rmse_te
    wandb.log({"Final_RMSE_validation": rmse_vl_min})
    wandb.log({"Final_RMSE_test": rmse_te_min})
    # Used for hyperparameter optimization, do not change the "RMSE" string
    print('RMSE',rmse_vl_min, rmse_te_min, flush=True)
    return rmse_vl_min, rmse_te_min

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
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
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()
    run(args)
