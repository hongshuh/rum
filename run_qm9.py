import os
import numpy as np
import torch
import dgl
import math
from rum.data import qm9,collate,qm9_xyz,qm9_graph
from rum.utils import Normalizer
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
def get_graphs(args,debug=False):

    # data = qm9
    if debug:
        train_set = qm9_graph('small_train',args)
        valid_set = qm9_graph('small_valid',args)
        test_set = qm9_graph('small_test',args)
    else:
        train_set = qm9_graph('train',args)
        valid_set = qm9_graph('valid',args)
        test_set = qm9_graph('test',args)

    return train_set, valid_set, test_set


def run(args):
    import yaml
    from datetime import datetime
    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    tmp_dir = f'/scratch/venkvis_root/venkvis/hongshuh/rum/checkpoint/qm9/{args.label}/{timestamp}'
    os.makedirs(tmp_dir,exist_ok=True)
    
    args_dict = vars(args)
    with open(f'{tmp_dir}/config.yaml', 'w') as file:
        yaml.dump(args_dict, file)

    accelerator = Accelerator(log_with="wandb",project_dir=tmp_dir)
    accelerator.init_trackers(project_name=f"RUM_qm9", config=args)
    device = accelerator.device
    if args.label == 'all':
        args.label = ['mu','alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv']
    else:
        args.label = args.label.split(',')
    train_set, valid_set, test_set = get_graphs(args,args.debug)
    accelerator.print("Train set size", len(train_set))
    accelerator.print("Valid set size", len(valid_set))
    accelerator.print("Test set size", len(test_set))
    
    normalizer = None
    if args.normalize_label == True:
        train_label = torch.tensor(np.array([train_set.df[l] for l in args.label]).T,requires_grad=False,dtype=torch.float32)
        normalizer = Normalizer(train_label)
        accelerator.print("Normalizing label, mean and std are",normalizer.mean, normalizer.std)
        accelerator.print("Train label shape",train_label.shape)
    
    data_train = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=8, collate_fn=collate
    )

    data_valid = dgl.dataloading.GraphDataLoader(
        valid_set, batch_size=2*args.batch_size,num_workers=8, collate_fn=collate
    )

    data_test = dgl.dataloading.GraphDataLoader(
        test_set, batch_size=2*args.batch_size,num_workers=8, collate_fn=collate
    )

    g, y ,walks,eids = next(iter(data_train))
    


    from rum.models import RUMGraphRegressionModel
    model = RUMGraphRegressionModel(
        in_features=g.ndata["h0"].shape[-1],
        out_features=len(args.label),
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
    ## Load checkpoint
    model = model.to(device)
    

################## Optimizer and Scheduler ##################
    
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

    steps_per_epoch = math.ceil(len(train_set) // args.batch_size)
    training_steps = steps_per_epoch * args.n_epochs
    if args.warmup_ratio > 1:
        warmup_steps = int(steps_per_epoch * args.warmup_ratio)
    else:
        warmup_steps = int(training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
    
    normalizer = accelerator.prepare(normalizer)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    data_train, data_valid, data_test = accelerator.prepare(data_train, data_valid, data_test)
################## Metrics ##################
    # Dummy MAE
    mae_vl, mae_te = torch.inf, torch.inf
    mae_vl_min = mae_vl
    mae_te_min = mae_te
    
################## Checkpointing ##################
    if args.checkpoint:
        accelerator.print("Loading checkpoint")
        try:
            accelerator.load_state(f"{args.checkpoint}")
            accelerator.print("Loading checkpoint successful")
        except:
            accelerator.print("Loading checkpoint failed")
    else:
        accelerator.print("Train from scratch")
        accelerator.save_state(f"{tmp_dir}/last",safe_serialization=False)
################## Training ##################
    for idx in range(args.n_epochs):
        model.train()
        for g, y ,walks,eids in data_train:
            optimizer.zero_grad()
            ## Forward
            h, loss = model(g, g.ndata["h0"], e=g.edata["e0"],walks=walks,eids=eids)
            if normalizer:
                y = normalizer.norm(y)
            mae = torch.nn.functional.l1_loss(h, y)
            loss = loss + mae
            ## Step
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            ## Log
            accelerator.log({"train loss": loss.item()})
            accelerator.log({"learning_rate": scheduler.get_last_lr()[0]})

        with torch.no_grad():
            model.eval()
            all_preds = []
            all_targets = []
            for g, y, walks, eids in data_valid:
                h_vl, _ = model(g, g.ndata["h0"], e=g.edata["e0"],walks=walks,eids=eids)
                if normalizer:
                    h_vl = normalizer.denorm(h_vl)
                pred,target = accelerator.gather_for_metrics((h_vl,y))
            #cat the predictions
                all_preds.append(pred)
                all_targets.append(target)
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            if idx == 0:
                accelerator.print('shape of all pred in valid:',all_preds.size())
            mae_vl = torch.mean(torch.abs(all_preds - all_targets), dim=0).cpu()
            for i,key in enumerate(args.label):
                accelerator.log({f"MAE_validation_{key}": mae_vl[i]})
      
            all_preds = []
            all_targets = []
            for g, y, walks, eids in data_test:
                h_te, _ = model(g, g.ndata["h0"], e=g.edata["e0"],walks=walks,eids=eids)
                if normalizer:
                    h_te = normalizer.denorm(h_te)
                pred,target = accelerator.gather_for_metrics((h_te,y))
                all_preds.append(pred)
                all_targets.append(target)
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            if idx == 0:
                accelerator.print('shape of all pred in test :',all_preds.size())
            mae_te = torch.mean(torch.abs(all_preds - all_targets), dim=0).cpu()
            for i,key in enumerate(args.label):
                accelerator.log({f"MAE_test_{key}": mae_te[i]})

            if torch.mean(mae_vl) < mae_vl_min:
                mae_te_min = torch.mean(mae_te)
                mae_vl_min = torch.mean(mae_vl)
                accelerator.wait_for_everyone()
                accelerator.save_state(f"{tmp_dir}/best",safe_serialization=False)
                for i,key in enumerate(args.label):
                    accelerator.log({f"Final_MAE_test_{key}": mae_te[i]})
        accelerator.wait_for_everyone()
        accelerator.save_state(f"{tmp_dir}/last",safe_serialization=False)
        accelerator.print(f"Epoch {idx} : MAE validation {torch.mean(mae_vl)}, MAE test {torch.mean(mae_te)}")

    accelerator.end_training()
    accelerator.print('MAE',mae_vl_min, mae_te_min, flush=True)
    return mae_vl_min, mae_te_min


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
    run(args)
