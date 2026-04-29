import os, random, pickle, numpy as np, torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----------------------------------------------------------------------
# Reproducibility Setup
# ----------------------------------------------------------------------
def set_seed(new_seed):
    """Seed"""
    os.environ['PYTHONHASHSEED'] = str(new_seed)
    random.seed(new_seed)
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)
    # cuDNN (CUDA Deep Neural Network library)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed fixed: {new_seed}]")
    return new_seed

seed = 45
set_seed(seed)

# ----------------------------------------------------------------------
# Paths and Input Data Loading
# ----------------------------------------------------------------------
current_script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_script_path))
raw_data_dir = os.path.join(parent_dir, 'data', 'Example_SFINCS_data.npz')

normalization_params_dir = os.path.join(parent_dir, 'data', 'normalization_params_minmax.npz')
os.makedirs(os.path.dirname(normalization_params_dir), exist_ok=True)
print(normalization_params_dir)

_loaded = np.load(raw_data_dir, allow_pickle=True)
all_data = {k: _loaded[k] for k in _loaded.files}

for name, arr in all_data.items():
    print(f"{name:16s} shape={arr.shape}, dtype={arr.dtype}")

def add_ch(x):
    if x.ndim == 2:
        return x[..., None]          # (H,W) → (H,W,1)
    elif x.ndim == 3:
        return np.moveaxis(x, 0, -1) # (T,H,W) → (H,W,T)
    else:
        raise ValueError("Unsupported ndim")

T = int(all_data["water_depth"].shape[0])
start, end = 0, T
t_min = start + 3
delta_t = 1  
t_max = end - delta_t - 1
if t_min > t_max:
    raise ValueError(
        f"Example NPZ: Time series length T={T}, delta_t={delta_t} is invalid (t_min={t_min}, t_max={t_max})."
    )

train_inputs, train_outputs = [], []

# ----------------------------------------------------------------------
# Training Sample Construction (Input/Target Pairing)
# ----------------------------------------------------------------------
for t in range(t_min, t_max + 1):
    tp = t + delta_t

    depth_win = add_ch(all_data["water_depth"][t - 3 : t + 1, :, :])
    curu_win = add_ch(all_data["current_u"][t - 3 : t + 1, :, :])
    curv_win = add_ch(all_data["current_v"][t - 3 : t + 1, :, :])
    rain_tp = add_ch(all_data["rainfall"][tp, :, :])
    wu_tp = add_ch(all_data["wind_u"][tp, :, :])
    wv_tp = add_ch(all_data["wind_v"][tp, :, :])
    prs_tp = add_ch(all_data["pressure"][tp, :, :])
    dis_tp = add_ch(all_data["discharge"][tp, :, :])
    zb_ch = add_ch(all_data["bathymetry"])
    n_ch = add_ch(all_data["manning_n"])
    x_list = [depth_win, curu_win, curv_win, rain_tp, wu_tp, wv_tp, prs_tp, dis_tp, zb_ch, n_ch]

    X = np.concatenate(x_list, axis=-1).astype(np.float32)

    y_depth = add_ch(all_data["water_depth"][tp, :, :])
    y_u = add_ch(all_data["current_u"][tp, :, :])
    y_v = add_ch(all_data["current_v"][tp, :, :])
    y = np.concatenate([y_depth, y_u, y_v], axis=-1).astype(np.float32)

    train_inputs.append(X)
    train_outputs.append(y)

X_train = np.stack(train_inputs, axis=0)
Y_train = np.stack(train_outputs, axis=0)
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)

# ----------------------------------------------------------------------
# Train data based on Min-Max normalization
# ----------------------------------------------------------------------
input_min = np.min(X_train, axis=(0, 1, 2))
input_max = np.max(X_train, axis=(0, 1, 2))
output_min = np.min(Y_train, axis=(0, 1, 2))
output_max = np.max(Y_train, axis=(0, 1, 2))

np.savez(
    normalization_params_dir,
    input_min=input_min,
    input_max=input_max,
    output_min=output_min,
    output_max=output_max,
)
print("\nnormalization_params_minmax.npz saved")


def normalize_minmax(x, vmin, vmax, eps=1e-12, clip=True):
    denom = vmax - vmin
    denom_safe = np.where(denom == 0, eps, denom)
    out = (x - vmin) / denom_safe
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)

X_train_n = normalize_minmax(X_train, input_min, input_max)
Y_train_n = normalize_minmax(Y_train, output_min, output_max)
print("  X_train_n:", X_train_n.shape, X_train_n.dtype)
print("  Y_train_n:", Y_train_n.shape, Y_train_n.dtype)

# ----------------------------------------------------------------------
# Training Pipeline Imports
# ----------------------------------------------------------------------
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(parent_dir)
from utils.SSWE_FDM_loss import compute_batch_momentum_loss
from utils.util import save, bestsave, load

# ----------------------------------------------------------------------
# Deep learning Models
# ----------------------------------------------------------------------
from models.UNet import UNet
from models.ConvLSTM import ConvLSTM
from models.SwinUNETR import SwinUNETR
from models.FNO import FNO

# ----------------------------------------------------------------------
# Hyperparameters and Loss Weights
# ----------------------------------------------------------------------
lr = 1e-3
batch_size = 6
num_epoch = 10

# loss weight initialization
data_loss_weight = 1
x_loss_weight = 1
y_loss_weight = 1
mass_loss_weight = 1

# ----------------------------------------------------------------------
# DWA and Warm-up Configuration (Liu et al., 2019)
# ----------------------------------------------------------------------
use_data = float(data_loss_weight) > 0
use_x = float(x_loss_weight) > 0
use_y = float(y_loss_weight) > 0
use_mass = float(mass_loss_weight) > 0

active_tasks = []
if use_data:
    active_tasks.append("data")
if use_mass:
    active_tasks.append("mass")
if use_x:
    active_tasks.append("x")
if use_y:
    active_tasks.append("y")

num_tasks = len(active_tasks)
T = 2.0  # DWA temperature
warmup_epochs = 10

# ----------------------------------------------------------------------
# Runtime Device and Output Paths
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# outputs / checkpoint / log 경로
outputs_dir = os.path.join(parent_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
Unique_name = "Example"
ckpt_dir = os.path.join(outputs_dir, f"checkpoint_{Unique_name}")
my_log_file = os.path.join(outputs_dir, f"log_{Unique_name}.pkl")
n_gpu = torch.from_numpy(all_data["manning_n"]).float().to(device) # Manning n

# ----------------------------------------------------------------------
# Tensor Conversion and DataLoader Setup
# ----------------------------------------------------------------------
X_train_t = torch.from_numpy(X_train_n).permute(0, 3, 1, 2).contiguous().float()
Y_train_t = torch.from_numpy(Y_train_n).permute(0, 3, 1, 2).contiguous().float()

dataset_train = TensorDataset(X_train_t, Y_train_t)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# ----------------------------------------------------------------------
# Min-Max Tensors and Physics Inputs
# ----------------------------------------------------------------------
input_min_gpu = torch.tensor(input_min, dtype=torch.float32, device=device).view(1, -1, 1, 1)
input_max_gpu = torch.tensor(input_max, dtype=torch.float32, device=device).view(1, -1, 1, 1)
output_min_gpu = torch.tensor(output_min, dtype=torch.float32, device=device).view(1, -1, 1, 1)
output_max_gpu = torch.tensor(output_max, dtype=torch.float32, device=device).view(1, -1, 1, 1)

# ----------------------------------------------------------------------
# Network, Loss Function, and Optimizer
# ----------------------------------------------------------------------
net = UNet().to(device)
fn_loss = nn.MSELoss().to(device)
optim = torch.optim.AdamW(net.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-4)

# ----------------------------------------------------------------------
# Checkpoint Resume and Training Logs
# ----------------------------------------------------------------------
st_epoch = 0
best_train = float("inf")
best_epoch = -1
net, optim, st_epoch, _, _ = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

loss_log = {
    "epoch": [],
    "train": [],
    "train_data_loss": [],
    "train_mass_loss": [],
    "train_x_loss": [],
    "train_y_loss": [],
    "learning_rate": [],
}
if os.path.exists(my_log_file):
    with open(my_log_file, "rb") as f:
        loss_log = pickle.load(f)

for epoch in range(st_epoch + 1, num_epoch + 1):
    # ------------------------------------------------------------------
    # Epoch Training Loop
    # ------------------------------------------------------------------
    net.train()
    current_lr = optim.param_groups[0]["lr"]
    loss_log["learning_rate"].append(current_lr)

    # ----------------------------
    # Dynamic Weight Averaging (DWA)
    # ----------------------------
    if epoch >= 3 and num_tasks > 1:
        key_map = {
            "data": "train_data_loss",
            "mass": "train_mass_loss",
            "x": "train_x_loss",
            "y": "train_y_loss",
        }

        prev_loss = np.array([loss_log[key_map[t]][-1] for t in active_tasks], dtype=np.float64)
        prev2_loss = np.array([loss_log[key_map[t]][-2] for t in active_tasks], dtype=np.float64)

        w = prev_loss / (prev2_loss + 1e-8)
        z_overflow = w / T
        z_overflow = z_overflow - np.max(z_overflow)
        w = np.exp(z_overflow)
        dwa_weights = num_tasks * w / (np.sum(w) + 1e-12)

        boost_factor = 1
        if "data" in active_tasks:
            data_idx = active_tasks.index("data")
            dwa_weights[data_idx] *= boost_factor
            dwa_weights = num_tasks * dwa_weights / np.sum(dwa_weights)

        new_w = dict(zip(active_tasks, dwa_weights.tolist()))
        data_loss_weight = float(new_w.get("data", 0.0))
        mass_loss_weight = float(new_w.get("mass", 0.0))
        x_loss_weight = float(new_w.get("x", 0.0))
        y_loss_weight = float(new_w.get("y", 0.0))

    # ----------------------------
    # warm-up (only for physics loss)
    # ----------------------------
    warmup_factor = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
    mass_w = mass_loss_weight * warmup_factor
    x_w = x_loss_weight * warmup_factor
    y_w = y_loss_weight * warmup_factor

    running_total = 0
    running_data = 0
    running_mass = 0
    running_x = 0
    running_y = 0

    for x_batch, y_batch in loader_train:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optim.zero_grad()
        pred = net(x_batch)

        data_loss = fn_loss(pred, y_batch)
        mass_loss, x_loss, y_loss = compute_batch_momentum_loss(
            input=x_batch,
            output=pred,
            input_min_gpu=input_min_gpu,
            input_max_gpu=input_max_gpu,
            output_min_gpu=output_min_gpu,
            output_max_gpu=output_max_gpu,
            n_gpu=n_gpu,
        )

        total_loss = (
            data_loss_weight * data_loss
            + mass_w * mass_loss
            + x_w * x_loss
            + y_w * y_loss
        )
        total_loss.backward()
        optim.step()

        running_total += total_loss.item()
        running_data += data_loss.item()
        running_mass += mass_loss.item()
        running_x += x_loss.item()
        running_y += y_loss.item()

    num_batch_train = max(1, len(loader_train))
    train_mean = running_total / num_batch_train
    train_data_mean = running_data / num_batch_train
    train_mass_mean = running_mass / num_batch_train
    train_x_mean = running_x / num_batch_train
    train_y_mean = running_y / num_batch_train

    loss_log["epoch"].append(epoch)
    loss_log["train"].append(train_mean)
    loss_log["train_data_loss"].append(train_data_mean)
    loss_log["train_mass_loss"].append(train_mass_mean)
    loss_log["train_x_loss"].append(train_x_mean)
    loss_log["train_y_loss"].append(train_y_mean)

    print(
        f"Epoch {epoch:04d} | "
        f"train={train_mean:.6e} "
        f"(data={train_data_mean:.6e}, mass={train_mass_mean:.6e}, "
        f"x={train_x_mean:.6e}, y={train_y_mean:.6e}) | "
        f"lr={current_lr:.2e} | "
        f"w(data={data_loss_weight:.3f}, mass={mass_w:.3f}, x={x_w:.3f}, y={y_w:.3f}) | "
        f"warmup={warmup_factor:.3f}"
    )

    # checkpoint save (train based)
    if train_mean < best_train:
        best_train = train_mean
        best_epoch = epoch
        bestsave(ckpt_dir=ckpt_dir, net=net, optim=optim)
        print(f"  -> BEST model updated: epoch={best_epoch}, train={best_train:.6e}")

    save(
        ckpt_dir=ckpt_dir,
        net=net,
        optim=optim,
        epoch=epoch,
        best_val=best_train,
        best_epoch=best_epoch,
    )

    with open(my_log_file, "wb") as f:
        pickle.dump(loss_log, f)

print(f"Training completed | best_epoch={best_epoch}, best_train={best_train:.6e}")


