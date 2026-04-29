import os 
import gc
import glob
import numpy as np
import torch
import torch.nn as nn


def safe_state_dict(model):
    """Move model state_dict tensors to CPU."""
    return {k: v.cpu() for k, v in model.state_dict().items()}

def save(ckpt_dir, net, optim, epoch, best_val, best_epoch, keep_last=2):
    """
    Save checkpoint every epoch and keep only the latest `keep_last` files.
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    net_cpu = safe_state_dict(net)
    optim_cpu = optim.state_dict()

    ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth")

    torch.save(
        {
            "net": net_cpu,
            "optim": optim_cpu,
            "epoch": epoch,
            "best_val": best_val,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )

    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "model_epoch*.pth")),
        key=os.path.getmtime,
    )

    if len(ckpt_files) > keep_last:
        to_delete = ckpt_files[:-keep_last]
        for f in to_delete:
            try:
                os.remove(f)
                #print(f"Deleted old checkpoint: {f}")
            except Exception as e:
                print(f"⚠️ Failed to delete {f}: {e}")

    del net_cpu, optim_cpu
    gc.collect()
    torch.cuda.empty_cache()


def finetuning_save(ckpt_dir, net, optim, epoch, best_val, best_epoch):
    """
    Save checkpoint for fine-tuning without deleting old checkpoints.
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    net_cpu = safe_state_dict(net)
    optim_cpu = optim.state_dict()

    ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth")

    torch.save(
        {
            "net": net_cpu,
            "optim": optim_cpu,
            "epoch": epoch,
            "best_val": best_val,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )

    del net_cpu, optim_cpu
    gc.collect()
    torch.cuda.empty_cache()


def bestsave(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    net_cpu = safe_state_dict(net)
    optim_cpu = optim.state_dict()

    torch.save({"net": net_cpu, "optim": optim_cpu}, os.path.join(ckpt_dir, "model_BEST.pth"))


    del net_cpu, optim_cpu
    gc.collect()
    torch.cuda.empty_cache()    


def load(ckpt_dir, net, optim): 
    if not os.path.exists(ckpt_dir):
        epoch = 0
        best_val = float("inf")
        best_epoch = -1
        return net, optim, epoch, best_val, best_epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if any(ch.isdigit() for ch in f)]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) 

    if len(ckpt_lst) == 0:
        epoch = 0
        best_val = float("inf")
        best_epoch = -1
        return net, optim, epoch, best_val, best_epoch    

    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]), weights_only=False)

    net.load_state_dict(dict_model["net"])
    optim.load_state_dict(dict_model["optim"])
    epoch = int(ckpt_lst[-1].split("epoch")[1].split(".pth")[0])

    best_val = dict_model.get("best_val", float("inf"))
    best_epoch = dict_model.get("best_epoch", -1)

    return net, optim, epoch, best_val, best_epoch

def bestload(ckpt_dir, net, optim):
    best_path = os.path.join(ckpt_dir, "model_BEST.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"BEST model not found at {best_path}")

    dict_model = torch.load(best_path, map_location="cpu", weights_only=False)

    net.load_state_dict(dict_model["net"])
    optim.load_state_dict(dict_model["optim"])

    epoch = dict_model.get("epoch", -1)
    best_val = dict_model.get("best_val", float("inf"))
    best_epoch = dict_model.get("best_epoch", -1)

    return net, optim, epoch, best_val, best_epoch
