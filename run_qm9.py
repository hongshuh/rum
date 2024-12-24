import numpy as np
from sqlalchemy import all_
import torch
import dgl
import math
from rum.data import qm9
from rum.utils import Normalizer
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)

def get_graphs(label='gap',debug=False):
    if debug:
        train_set = qm9('small_train',label)
        valid_set = qm9('small_valid',label)
        test_set = qm9('small_test',label)
    else:
        train_set = qm9('train',label)
        valid_set = qm9('valid',label)
        test_set = qm9('test',label)

    return train_set, valid_set, test_set


def run(args):
    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device
    train_set, valid_set, test_set = get_graphs(args.label)

    normalizer = None
    if args.normalize_label == True:
        train_label = torch.tensor(train_set.df[train_set.label].values,dtype=torch.float32,requires_grad=False)
        normalizer = Normalizer(train_label)
        accelerator.print("Normalizing label, mean and std are",normalizer.mean, normalizer.std)
        accelerator.print("Train label shape",train_label.shape)

    data_train = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    data_valid = dgl.dataloading.GraphDataLoader(
        valid_set, batch_size=2*args.batch_size,
    )

    data_test = dgl.dataloading.GraphDataLoader(
        test_set, batch_size=2*args.batch_size,
    )

    g, y = next(iter(data_train))
    
    ## Wandb Init
    import wandb
    from datetime import datetime

    # Get current date and time
    now = datetime.now()

    # Format date and time to a string (e.g., "2024-12-21_14-30-45")
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Print or use the timestamp as a folder name
    accelerator.print(timestamp)
    accelerator.init_trackers(project_name=f"RUM_{args.data}_{args.label}", config=args,init_kwargs={"wandb": {"group": f"{timestamp}"}})

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
    model = model.float()

    ## Count number of parameters in Million
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Number of parameters : {np.round(total_params/1e6,1)} M")

    model = model.to(device)
    

    
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
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    data_train, data_valid, data_test = accelerator.prepare(data_train, data_valid, data_test)

    from rum.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Dummy MAE
    mean_train = train_set.df[args.label].mean()
    mae_vl = valid_set.df[args.label].apply(lambda x: abs(x - mean_train)).mean()
    mae_te = test_set.df[args.label].apply(lambda x: abs(x - mean_train)).mean()
    mae_vl_min = mae_vl
    mae_te_min = mae_te
    accelerator.print(f'Dummy MAE on validation and test set : {mae_vl_min}, {mae_te_min}')
    
    for idx in range(args.n_epochs):
        for g, y in data_train:
            model.train()
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["h0"], e=g.edata["e0"])
            
            # mse = torch.nn.functional.mse_loss(h.squeeze(), y)
            if normalizer:
                y = normalizer.norm(y)
            mae = torch.nn.functional.l1_loss(h.squeeze(), y)
            loss = loss + mae
            
            # loss.backward()
            accelerator.backward(loss)
            
            # logging loss to wandb
            accelerator.log({"train loss": loss.item()})
            # wandb.log({"train MAE": mae.item()})
    
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            all_preds = []
            all_targets = []
            for g, y in data_valid:
                h_vl, _ = model(g, g.ndata["h0"], e=g.edata["e0"])
                if normalizer:
                    h_vl = normalizer.denorm(h_vl)
                pred,target = accelerator.gather_for_metrics((h_vl.squeeze(),y))
            #cat the predictions
                all_preds.append(pred)
                all_targets.append(target)
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            accelerator.print('shape of all pred = ',all_preds.size())
            mae_vl = torch.nn.functional.l1_loss(all_preds,all_targets).item()
            accelerator.log({"MAE_validation": mae_vl})
            if early_stopping([mae_vl]) and idx > 200:
                # print("Early stopping at epoch :", idx)
                break

            for g, y in data_test:
                all_preds = []
                all_targets = []
                h_te, _ = model(g, g.ndata["h0"], e=g.edata["e0"])
                if normalizer:
                    h_te = normalizer.denorm(h_te)
                pred,target = accelerator.gather_for_metrics((h_te.squeeze(),y))
                all_preds.append(pred)
                all_targets.append(target)
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            accelerator.print('shape of all pred = ',all_preds.size())
            mae_te = torch.nn.functional.l1_loss(all_preds,all_targets).item()
            accelerator.log({"MAE_test": mae_te})

            # print(rmse_vl, rmse_te)
            if mae_vl < mae_vl_min:
                mae_te_min = mae_te
                mae_vl_min = mae_vl
        
    accelerator.log({"Final_MAE_validation": mae_vl_min})
    accelerator.log({"Final_MAE_test": mae_te_min})
    accelerator.end_training()
    # Used for hyperparameter optimization, do not change the "RMSE" string
    accelerator.print('MAE',mae_vl_min, mae_te_min, flush=True)
    return mae_vl_min, mae_te_min

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="qm9")
    parser.add_argument("--label", type=str, default="gap")
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--normalize_label", type=bool, default=True)
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()
    run(args)
