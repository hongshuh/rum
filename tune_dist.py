from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
# from ray.tune.trainable import session
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
import torch
import os
import os
os.environ["RAY_TMP_DIR"] = "/scratch/venkvis_root/venkvis/hongshuh/rum"
# ray.init(num_cpus=os.cpu_count())
ray.init(num_gpus=0,num_cpus=2, _temp_dir='/scratch/venkvis_root/venkvis/hongshuh/rum')
# LSF_COMMAND = "bsub -q gpuqueue -gpu " +\
# "\"num=1:j_exclusive=yes\" -R \"rusage[mem=5] span[ptile=1]\" -W 0:10 -Is "

"""Use absolute path"""
PYTHON_COMMAND =\
"python /nfs/turbo/coe-venkvis/hongshuh/rum/run.py"

"""Don't use sbatch"""
# SLURM_COMMAND = (
#     "sbatch --job-name=rum_tune_esol "
#     "--nodes=1 --partition=venkvis-h100 "
#     "--gres=gpu:h100:1 --cpus-per-task=2 "
#     "--time=1:00:00 --mem=10G "
#     "--output=log/rum_tune_esol_tune.out "
#     "--error=log/rum_tune_esol_tune.err "
#     "--mail-user=changwenxu1998@gmail.com "
#     "--mail-type=ALL "
# )

# def args_to_command(args):
#     command = SLURM_COMMAND + "--wrap=\"" + PYTHON_COMMAND
#     for key, value in args.items():
#         command += f" --{key} {value}"
#     command += "\""
#     return command

"""Submit interact job"""
# SLURM_COMMAND = (
#     "srun --job-name=rum_tune_esol "
#     "--nodes=1 --partition=venkvis-h100 "
#     "--gres=gpu:h100:1 --cpus-per-task=2 "
#     "--time=0:04:00 --mem=10G "
# )
SLURM_COMMAND = (
    "srun --job-name=rum_tune_esol "
    "--nodes=1 --partition=venkvis-h100 "
    "--gres=gpu:h100:1 --cpus-per-task=1 "
    "--time=0:20:00 --mem=10G "
)

def args_to_command(args):
    command = SLURM_COMMAND + " python /nfs/turbo/coe-venkvis/hongshuh/rum/run.py"
    for key, value in args.items():
        command += f" --{key} {value}"
    return command

def lsf_submit(command):
    import subprocess
    print("--------------------")
    print("Submitting command:")
    print(command)
    print("--------------------")
    output = subprocess.getoutput(command)
    return output

# def parse_output(output):
#     line = output.split("\n")[-1]
#     if "ACCURACY" not in line:
#         print(output, flush=True)
#         return 0.0, 0.0
#     _, accuracy_vl, accuracy_te = line.split(",")
#     return float(accuracy_vl), float(accuracy_te)

def parse_output(output):
    line = output.split("\n")[-1]
    try:
        rmse_vl, rmse_te = map(float, line.split())
    except ValueError:
        print("Failed to parse output:", output, flush=True)
        return float('inf'), float('inf')  # Returning high values if parsing fails
    return rmse_vl, rmse_te

def objective(args):
    # checkpoint = os.path.join(os.getcwd(), "model.pt")
    # args["checkpoint"] = checkpoint
    command = args_to_command(args)
    output = lsf_submit(command)
    rmse_vl_min, rmse_te_min = parse_output(output)
    # train.report({"accuracy": accuracy, "accuracy_te": accuracy_te})
    train.report({"rmse_vl_min": rmse_vl_min, "rmse_te_min": rmse_te_min})
    # session.report({"accuracy": accuracy, "accuracy_te": accuracy_te})
    # print(f"accuracy: {accuracy}")
    # print(f"accuracy_te: {accuracy_te}")
    print(f"rmse_vl_min: {rmse_vl_min}")
    print(f"rmse_te_min: {rmse_te_min}")

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(64, 128),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "length": tune.randint(8, 16),
        "temperature": tune.uniform(0.0, 1.0),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "AdamW",
        "activation": "SiLU",
        "depth": tune.randint(1, 3),
        # "num_layers": tune.randint(1, 16),
        "num_samples": tune.randint(8, 32),
        "n_epochs": 1000,
        "self_supervise_weight": tune.loguniform(1e-4, 1e-1),
        "consistency_weight": tune.loguniform(1e-4, 1e-1),
        "dropout": 0.1,
    }
    # param_space = {
    #     "data": args.data,
    #     "hidden_features": 64,
    #     "learning_rate": 1e-3,
    #     "weight_decay": 1e-4,
    #     "length": tune.randint(4, 16),
    #     "learning_rate": 1e-3,
    #     "temperature": tune.uniform(0.0, 1.0),
    #     "consistency_temperature": tune.uniform(0.0, 1.0),
    #     "optimizer": "Adam",
    #     "depth": 2,
    #     "num_layers": 3,
    #     "num_samples": 8,
    #     "n_epochs": 1000,
    #     "self_supervise_weight": tune.loguniform(1e-4, 1e-1),
    #     "consistency_weight": tune.loguniform(1e-4, 1e-1),
    #     "dropout": 0.1,
    # }
    tune_config = tune.TuneConfig(
        # metric="_metric/accuracy",
        metric="rmse_te_min", 
        mode="min",
        search_alg=OptunaSearch(),
        num_samples=200,
    )
    run_config = air.RunConfig(
        name=name,
        # storage_path=args.data,
        verbose=1,
    )
    tuner = tune.Tuner(
        objective,
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    args = parser.parse_args()
    experiment(args)
