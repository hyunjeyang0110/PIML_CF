#!/usr/bin/env python
# coding: utf-8

# --------------------- Super computer version ---------------------
'''
기준: PIML or Data
1) PIML or Data only
2) FDM, FVM etc. 
3) DWA boost factor
4. Delta t
'''

delta_t = 1  # ex: N시간 후 시점을 출력으로 지정; 12이면 12시간 이후 시점 

# Storm: 1. Rita(2005) 2. Humberto(2007) 3. Edouard (2008) 4. Ike(2008) 5. Harvey(2017) 6. Imelda(2019) 7. Beta(2020)
# ⚠️ 이제는 7개 허리케인 모두 사용합니다 (제외 없음)
# 원본 7개 이벤트 번호 기준: 1. Rita, 2. Humberto, 3. Edouard, 4. Ike, 5. Harvey, 6. Imelda, 7. Beta
# holdout_event_order = None  # 제외할 이벤트 없음 (모두 사용)
val_event_order = 6      # 원본 7개 중 6번째 = Imelda
test_event_order = 4     # 원본 7개 중 4번째 = Ike

# =====================================================
# 🔒 Random seed (for reproducibility)
# =====================================================
import os, random, numpy as np, torch
import sys

def set_seed(new_seed):
    """Seed를 설정하는 함수"""
    os.environ['PYTHONHASHSEED'] = str(new_seed)
    random.seed(new_seed)
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)  # multi-GPU 환경 대비
    # cuDNN (CUDA Deep Neural Network library)을 deterministic 모드로 (완전한 재현성 확보)
    torch.backends.cudnn.deterministic = True # 항상 같은 커널, 같은 연산 순서로 실행하라는 뜻. 
    torch.backends.cudnn.benchmark = False # 기본적으로 True이면 cuDNN이 실행마다 "benchmarking"을 해서 빠른 커널을 찾는데, 이게 바로 비결정적 동작의 원인 중 하나
    print(f"[Seed fixed: {new_seed}]")
    return new_seed

# 초기 seed 설정 (a_sbatch_FDM.py에서 변경 가능)
seed = 99999  # 초기 seed는 Unique_number에 항상 포함됨
initial_seed = seed  # a_sbatch_FDM.py에 의해 변경된 seed 값을 initial_seed로 저장

# 재시작 횟수 확인 (환경변수에서 읽기)
restart_count = 0
if 'RESTART_COUNT' in os.environ:
    restart_count = int(os.environ['RESTART_COUNT'])
    print(f"[재시작 감지] 재시작 횟수: {restart_count}")

# 재시작 시 seed 확인 (환경변수에서 읽기)
restart_seed = None
if 'RESTART_SEED' in os.environ:
    restart_seed = int(os.environ['RESTART_SEED'])
    seed = restart_seed  # 실제 학습에 사용할 seed는 재시작 seed
    print(f"[재시작 감지] 환경변수에서 seed 읽음: {seed}")
    del os.environ['RESTART_SEED']  # 다음 재시작을 위해 삭제

# Seed 정보 출력 (디버깅용)
print(f"[Seed 정보] 초기 seed (Unique_number에 사용): {initial_seed}")
print(f"[Seed 정보] 실제 학습에 사용할 seed: {seed}")

set_seed(seed)

# Unique_number 생성: 초기 seed는 항상 포함, 재시작 seed가 있으면 뒤에 추가
if restart_seed is not None:
    Unique_number = "Ablation_dt" + str(delta_t) + "_v" + str(val_event_order) + "t" + str(test_event_order) + "_seed" + str(initial_seed) + "_" + str(restart_seed)
else:
    Unique_number = "Ablation_dt" + str(delta_t) + "_v" + str(val_event_order) + "t" + str(test_event_order) + "_seed" + str(initial_seed)
print("Unique number:", Unique_number)

# directories / 이 모든 데이터는 supercomputer에서는 work directory로 넣어준다. 
import os
raw_data_dir = r"/work/09441/yanghj2002/ls6/PINN/data/Raw_data_sfincs_results.pkl"
print(raw_data_dir)

data_dir = r"/work/09441/yanghj2002/ls6/PINN/data/" + Unique_number
os.makedirs(data_dir, exist_ok=True)
print(data_dir)

normalization_params_dir = os.path.join(data_dir, 'normalization_params_minmax.npz')
os.makedirs(os.path.dirname(normalization_params_dir), exist_ok=True)
print(normalization_params_dir)



import pickle
import pandas as pd

'''
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
251027에 생성한 전체 Raw data이다. 
dictionary형태로 되어있으며 sfincs_data에서 추출한 모든 output data가 포함되어있다. 
데이터는 7개의 hurricane이 모두 포함되어있으며 그 길이는 각각 [82, 64, 40, 70, 250, 88, 160] 이다. 
여기서는 Raw data를 불러온 이후에 input과 output에 내가 원하는 데이터들을 넣어주는 작업을 한다. 
'''

all_data = pd.read_pickle(raw_data_dir)

for name, arr in all_data.items():
    print(f"{name:16s} shape={arr.shape}, dtype={arr.dtype}")


import numpy as np
import pandas as pd

# === 이벤트 길이 및 구간 설정 ===
# 원본 7개 이벤트: [82, 64, 40, 70, 250, 88, 160]
# ⚠️ 이제는 7개 이벤트 모두 사용합니다 (제외 없음)
all_event_lengths = [82, 64, 40, 70, 250, 88, 160]
event_lengths = all_event_lengths  # 모든 이벤트 사용
print(f"✅ 7개 이벤트 모두 사용합니다. 사용할 이벤트 길이: {event_lengths}")

event_starts = np.cumsum([0] + event_lengths[:-1]).tolist()
event_ranges = [(s, s + L) for s, L in zip(event_starts, event_lengths)]

# === 유틸리티 함수 ===
def add_ch(x):
    if x.ndim == 2:
        return x[..., None]          # (H,W) → (H,W,1)
    elif x.ndim == 3:
        return np.moveaxis(x, 0, -1) # (T,H,W) → (H,W,T)
    else:
        raise ValueError("Unsupported ndim")

# === 입력/출력 생성 ===
inputs, outputs = [], []

event_lengths_final = []  # 각 허리케인별 최종 길이 저장 리스트

# 원본 7개 이벤트의 시간 구간 계산
all_event_starts = np.cumsum([0] + all_event_lengths[:-1]).tolist()
all_event_ranges = [(s, s + L) for s, L in zip(all_event_starts, all_event_lengths)]

# 7개 이벤트 모두 처리 (제외 없음)
for event_idx_0based, (start, end) in enumerate(all_event_ranges):
    t_min = start + 3
    t_max = end - delta_t - 1
    if t_min > t_max:
        continue

    event_inputs, event_outputs = [], []  # 각 허리케인별 임시 리스트

    for t in range(t_min, t_max + 1):
        tp = t + delta_t

        # === 입력 채널 구성 ===
        depth_win = add_ch(all_data["water_depth"][t-3:t+1, :, :])
        curu_win  = add_ch(all_data["current_u"][t-3:t+1, :, :])
        curv_win  = add_ch(all_data["current_v"][t-3:t+1, :, :])
        rain_tp   = add_ch(all_data["rainfall"][tp, :, :])
        wu_tp     = add_ch(all_data["wind_u"][tp, :, :])
        wv_tp     = add_ch(all_data["wind_v"][tp, :, :])
        prs_tp    = add_ch(all_data["pressure"][tp, :, :])
        dis_tp    = add_ch(all_data["discharge"][tp, :, :])
        zb_ch     = add_ch(all_data["bathymetry"])
        n_ch      = add_ch(all_data["manning_n"])
        x_list = [depth_win, curu_win, curv_win, rain_tp, wu_tp, wv_tp, prs_tp, dis_tp, zb_ch, n_ch]

        if delta_t > 1: # 2 이상부터 평균값을 넣어준다. 
            rain_avg  = add_ch(np.mean(all_data["rainfall"][t+1:tp+1, :, :], axis=0))
            wu_avg    = add_ch(np.mean(all_data["wind_u"][t+1:tp+1, :, :], axis=0))
            wv_avg    = add_ch(np.mean(all_data["wind_v"][t+1:tp+1, :, :], axis=0))
            prs_avg   = add_ch(np.mean(all_data["pressure"][t+1:tp+1, :, :], axis=0))
            dis_avg   = add_ch(np.mean(all_data["discharge"][t+1:tp+1, :, :], axis=0))
            x_list += [rain_avg, wu_avg, wv_avg, prs_avg, dis_avg]

        X = np.concatenate(x_list, axis=-1).astype(np.float32)

        # === 출력 채널 구성 ===
        y_depth = add_ch(all_data["water_depth"][tp, :, :])
        y_u     = add_ch(all_data["current_u"][tp, :, :])
        y_v     = add_ch(all_data["current_v"][tp, :, :])
        y = np.concatenate([y_depth, y_u, y_v], axis=-1).astype(np.float32)

        event_inputs.append(X)
        event_outputs.append(y)

    # === ✅ 각 허리케인별 길이를 6의 배수로 맞추기 ===
    event_len = len(event_inputs)
    remain = event_len % 6
    if remain != 0:
        event_inputs = event_inputs[:-remain]
        event_outputs = event_outputs[:-remain]
        print(f"Event truncated: {event_len} → {len(event_inputs)} (removed {remain})")

    # === 전체 리스트에 추가 ===
    inputs.extend(event_inputs)
    outputs.extend(event_outputs)

    # === ✅ 허리케인별 길이 저장 ===
    event_lengths_final.append(len(event_inputs))

    print(f"Event done | samples: {len(event_inputs)}")

# === 요약 출력 ===
print("\n=== Summary of Each Hurricane ===")
print(f"✅ 7개 이벤트 모두 사용됩니다.")
for i, L in enumerate(event_lengths_final, start=1):
    print(f"Event {i}: {L} samples")

print(f"\nTotal samples: {sum(event_lengths_final)}")
print("event_lengths_final:", event_lengths_final)



'''
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
원본 7개의 Hurricanes 모두 사용합니다 (제외 없음).
7개의 허리케인을 기준으로 training, validation, test dataset을 나눠줍니다.
원본 이벤트 번호: 1. Rita, 2. Humberto, 3. Edouard, 4. Ike (test), 5. Harvey, 6. Imelda (val), 7. Beta
'''

import numpy as np

inputs = np.array(inputs)
outputs = np.array(outputs)

print(inputs.shape)   # inputs:  (총 샘플 수, nx, ny, 24 or 19)
print(outputs.shape)  # outputs: (총 샘플 수, nx, ny, 3)

event_lengths = event_lengths_final   # length after trimming (t-3 to t+1 time span으로 나눈 이후의 상황)

def build_event_ranges(lengths):
    """각 이벤트가 inputs/outputs에서 차지하는 인덱스 범위(start, end_exclusive) 리스트."""
    starts = np.cumsum([0] + lengths[:-1])
    return [(int(s), int(s+L)) for s, L in zip(starts, lengths)]

event_ranges = build_event_ranges(event_lengths)
# 예: [(0,78), (78,141), (141,180), (180,249), (249,495), (495,582), (582,741)]

def indices_from_events(event_ranges, events_1based):
    """1-based 이벤트 번호 리스트를 받아 해당 샘플 인덱스 배열을 반환."""
    if not events_1based:
        return np.array([], dtype=np.int64)
    idx = []
    for e in events_1based:
        k = e - 1  # 0-based
        s, t = event_ranges[k]
        idx.extend(range(s, t))
    return np.array(idx, dtype=np.int64)

def simple_split_by_events(inputs, outputs, event_ranges,
                           val_events_1b=None, test_events_1b=None):
    """
    events를 1-based로 받음.
    지정 안 된 이벤트는 전부 train으로 자동 배정.
    """
    val_events_1b  = val_events_1b  or []
    test_events_1b = test_events_1b or []

    all_events = set(range(1, len(event_ranges)+1))
    val_set  = set(val_events_1b)
    test_set = set(test_events_1b)
    assert val_set.isdisjoint(test_set), "val과 test에 중복 이벤트가 있습니다."

    train_set = sorted(all_events - val_set - test_set)
    val_set   = sorted(val_set)
    test_set  = sorted(test_set)

    train_idx = indices_from_events(event_ranges, train_set)
    val_idx   = indices_from_events(event_ranges, val_set)
    test_idx  = indices_from_events(event_ranges, test_set)

    X_train, Y_train = inputs[train_idx], outputs[train_idx]
    X_val,   Y_val   = inputs[val_idx],   outputs[val_idx]
    X_test,  Y_test  = inputs[test_idx],  outputs[test_idx]

    print(f"Events -> train:{train_set}, val:{val_set}, test:{test_set}")
    print("Shapes:")
    print("  train:", X_train.shape, Y_train.shape)
    print("  val  :", X_val.shape,   Y_val.shape)
    print("  test :", X_test.shape,  Y_test.shape)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx)


'''
Example (Ike 제외 후 6개 이벤트 기준)
2,5 을 validation으로; 4를 test로 하는경우 / !!! 여기서는 1부터 6까지이다 !!!
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test), _ = \
    simple_split_by_events(inputs, outputs, event_ranges,
                           val_events_1b=[2,5], test_events_1b=[4])
'''

# --- Main code --- 
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (idx_tr, idx_val, idx_te) = \
    simple_split_by_events(inputs, outputs, event_ranges,
                           val_events_1b=[val_event_order], test_events_1b=[test_event_order])



'''
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Training dataset만을 기준으로 min-max scaling을 진행한다. 
또한 이때의 min, max value를 denorm이 가능하게 pkl file로 저장해 준다. 
'''

import numpy as np
import pickle

# inputs과 outputs은 학습 데이터셋 전체를 담고 있는 numpy 배열이라고 가정
# inputs.shape: (n_samples, 600, 600, 17)
# outputs.shape: (n_samples, 600, 600)

# inputs과 outputs의 평균 및 표준편차 계산
input_min = np.min(X_train, axis=(0, 1, 2))
input_max = np.max(X_train, axis=(0, 1, 2))

output_min = np.min(Y_train, axis=(0, 1, 2))
output_max = np.max(Y_train, axis=(0, 1, 2))

# 정규화 파라미터들을 딕셔너리로 묶기
normalization_params = {
    'input_min': input_min,
    'input_max': input_max,
    'output_min': output_min,
    'output_max': output_max
}

# npz 파일로 저장
np.savez(normalization_params_dir,
         input_min=input_min,
         input_max=input_max,
         output_min=output_min,
         output_max=output_max)

print("\n normalization_params save finish ")



# # 🌵 정규화 및 Data save
# ```
# 여기서는 train, val, test라는 파일을 만들어주면서 그 안에 내가 원하는대로 input, label 파일들을 만들어 준다. 
# 위에서 training set에서의 min, max value를 얻었기 때문에 이에 대한 정규화를 실시해 주어야 한다.
# 
# ```


import os

dir_save_train = os.path.join(data_dir, 'train')
dir_save_val = os.path.join(data_dir, 'val')
dir_save_test = os.path.join(data_dir, 'test')

# 디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)



# -------------------
# 정규화 함수 (train 통계 사용)
# -------------------
def normalize_minmax(x, vmin, vmax, eps=1e-12, clip=True):
    # vmin/vmax shape는 채널 축과 일치해야 함.
    # x: (..., C), vmin/vmax: (C,)
    denom = (vmax - vmin)
    # 분모가 0인 채널 방지
    denom_safe = np.where(denom == 0, eps, denom)
    out = (x - vmin) / denom_safe
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out

# -------------------
# 저장 유틸
# -------------------
def save_split(X, Y, out_dir, start_idx=0):
    """
    X: (N, H, W, Cin), Y: (N, H, W, Cout)
    파일명은 input_###.npy, label_###.npy 로 저장.
    start_idx는 번호 이어붙일 때 사용.
    """
    N = X.shape[0]
    for i in range(N):
        np.save(os.path.join(out_dir, f'input_{i+start_idx:03d}.npy'),  X[i])
        np.save(os.path.join(out_dir, f'label_{i+start_idx:03d}.npy'),  Y[i])
    return start_idx + N  # 다음 시작 번호 리턴

# -------------------
# 정규화 적용 (train 통계로 전체 정규화)
# -------------------
# 이미 계산한 값 사용: input_min, input_max, output_min, output_max
# shape 예: input_min/max -> (596,596,17) 아니고 (17,)이어야 브로드캐스팅이 직관적.
# axis=(0,1,2)로 계산했으면 (17,) / (3,)이 맞음.

import pickle
import numpy as np

#params = np.load(normalization_params_dir)
#input_min  = params['input_min']
#input_max  = params['input_max']
#output_min = params['output_min']
#output_max = params['output_max']

X_train_n = normalize_minmax(X_train, input_min,  input_max)  # (N, H, W, 17)
Y_train_n = normalize_minmax(Y_train, output_min, output_max) # (N, H, W, 3)

X_val_n   = normalize_minmax(X_val,   input_min,  input_max)
Y_val_n   = normalize_minmax(Y_val,   output_min, output_max)

X_test_n  = normalize_minmax(X_test,  input_min,  input_max)
Y_test_n  = normalize_minmax(Y_test,  output_min, output_max)

# -------------------
# 저장 (번호는 각 split에서 000부터 시작)
# -------------------
_ = save_split(X_train_n, Y_train_n, dir_save_train, start_idx=0) # 리턴값을 안쓰겠다는 것임. 
_ = save_split(X_val_n,   Y_val_n,   dir_save_val,   start_idx=0)
_ = save_split(X_test_n,  Y_test_n,  dir_save_test,  start_idx=0)

print("저장 완료:")
print("  train:", X_train_n.shape, Y_train_n.shape)
print("  val  :", X_val_n.shape,   Y_val_n.shape)
print("  test :", X_test_n.shape,  Y_test_n.shape)


# In[ ]:






# # Model training

# In[3]:


import sys, os
Main_dir = os.path.dirname(os.path.dirname(raw_data_dir))
sys.path.append(Main_dir) # Main dir에 있는 곳까지 포함해서 src.utils.dataset etc.의 Module을 불러올 수 있다. 
#os.chdir(Main_dir)

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets

import sys

from src.models.model import UNet
from src.utils.dataset import * # *는 모든 클래스를 불러오는 것
from src.utils.util import *
from src.losses.SSWE_FDM_loss import compute_batch_momentum_loss # FDM code

# 🎯 New DL model 
from src.models.ConvLSTM import FloodConvLSTMModel

# In[6]:


# 하이퍼 파라미터 설정
lr = 1e-3
batch_size = 6
num_epoch = 2000

import pickle
import numpy as np


# 각 변수 torch로 변환
input_min_gpu  = torch.tensor(input_min,  device="cuda").view(1, -1, 1, 1)
input_max_gpu   = torch.tensor(input_max,   device="cuda").view(1, -1, 1, 1)
output_min_gpu = torch.tensor(output_min, device="cuda").view(1, -1, 1, 1)
output_max_gpu  = torch.tensor(output_max,  device="cuda").view(1, -1, 1, 1)

# GPU, CPU를 정해주는거
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 내 환경에서 cuda가 지원 가능하다면 device가 cuda (GPU)가 되는거고, 아니라면 cpu를 하라는 것이다. 

# outputs 폴더 경로 (PINN/outputs)
outputs_dir = os.path.join(Main_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# checkpoint와 log 경로 설정
ckpt_dir = os.path.join(outputs_dir, f"checkpoint_{Unique_number}")
my_log_file = os.path.join(outputs_dir, f"log_{Unique_number}.pkl")

# Manning n loading
import pandas as pd
base_dir = os.path.dirname(data_dir)
n = pd.read_pickle(os.path.join(base_dir, 'Manning_600600.pkl'))
n = n[2:-2, 2:-2] # 상하좌우 2줄 제거 → 596x596 기본적으로 모두 596*596 체제이다. 
n_gpu = torch.from_numpy(n).float().to("cuda").view(596, 596)


# In[7]:


# Environment setting
transform = transforms.Compose([ToTensor()]) # input data pre-treatment를 3가지 요소를 결합해서 transform이라는 것으로 합쳐준다! # 여기서는 RandomFlip을 없애주었다. / 왜냐하면 training할 때도 batch의 순서가 그대로 들어와야 하기 때문이다. 

dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform) # 데이터 우선 불러오는데 transform을 활용해서 불러온다. 
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0) # 데이터를 그냥 가져오는게 아니라 학습 과정에 맞게 효율적이게 가져온다./ 학습 시 데이터를 한번에 batch_size개씩 묶어서 모델에 주게 된다. num_workers는 CPU N개에서 병렬처리를 한다는 것이다. 

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0) # 윈도우에서는 num_workers가 0이 되어야 하는 경우가 있다. 

# 네트워크 생성
net = FloodConvLSTMModel().to(device) # 내가 정의한 UNet 모델클래스의 객체 (인스턴스)를 생성하였다. 위에 device를 정의해 주었다. cpu또는 cuda에서 돌릴 준비를 한다는 것이다. 상황에 따라서 맞춘다. 

# loss function
fn_loss = nn.MSELoss().to(device)
#fn_loss = lambda yhat, y: torch.sqrt(nn.MSELoss()(yhat, y))

# Optimizer
#optim = torch.optim.AdamW(net.parameters(), lr=lr)
optim = torch.optim.AdamW(net.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-4) # weight decay는 lr와는 관련없는거다. 

scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=20) # validation loss기준으로 lr 감소시키는 tool 

# 부수적인 variables설정
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size) # 몇번 batch size가 돌아가야 하는지 알려준다. 
num_batch_val = np.ceil(num_data_val / batch_size)

# 부수적인 function / # 간단한 함수를 만들 때 쓰는 문법이다. // lambda 매개변수 : 리턴값 구조
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # 여기서는 CPU로 가져와서, 학습 그래프에서 detach하고 (gradient 추적X), torch tensor를 numpy배열로 변환한 이후, batch, channel, height, width 의 torch 순서를 batch, height, width, channel 로 바꿔주는 과정이다. 
    # 즉 numpy 이미지 형태로 바꿔주는 기능을 한다. 
fn_denorm = lambda x, mean, std: (x * std) + mean # Normalization한 데이터를 원래의 스케일로 되돌리는 함수. 
fn_denorm2 = lambda x, min, max: (x * (max-min)) + min # MinMax scaling에서 Normalization한 데이터를 원래의 스케일로 되돌리는 함수. 
fn_class = lambda x: 1 * (x>0.5)


# In[ ]:


import gc
import time

data_loss_weight = 1
x_loss_weight = 1
y_loss_weight = 0
mass_loss_weight = 0

# =====================================================
# ✅ Physics cap (data 우선 보장)
# - physics 가중치 합이 data 가중치의 일정 비율을 넘지 못하게 제한
# - DWA의 "physics 내부 비율"은 유지하되, 전체 physics 영향력만 상한을 둠
# =====================================================
DATA_W_MIN = 1.0          # data weight 최소값
PHYS_CAP_RATIO = (x_loss_weight + y_loss_weight + mass_loss_weight)/10  # (wx+wy+wm) <= PHYS_CAP_RATIO * wd

# =====================================================
# ✅ DWA에서 "활성화"할 loss 선택
# - (520~523)에서 weight를 0으로 두면 해당 loss는 DWA에서도 완전히 제외됨
# - weight > 0 인 loss들만 DWA로 가중치가 자동 조정됨
# =====================================================
use_data = float(data_loss_weight) > 0
use_x    = float(x_loss_weight) > 0
use_y    = float(y_loss_weight) > 0
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

# DWA task 개수 (기존에는 "가중치 합"을 썼지만, on/off를 위해 "활성 task 개수"로 변경)
num_tasks = len(active_tasks)
T = 2 # DWA temperature (보통 1.5~2.0 사이 사용)

warmup_epochs = 10 # physics loss부분을 점진적으로 올린다. ; 이거 우선 하지 말아보자 
early_stop_patience = 10000 # N번동안 validation이 증가하지 않으면 멈춘다. 개선판단 여유폭은 delta가 결정한다. 
patience_counter = 0


import wandb

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="hyunjeyang-the-university-of-texas-at-austin",
    # Set the wandb project where this run will be logged.
    project="PIML",
    name=Unique_number, 
    mode="online",
    config={
        "Unique_number": Unique_number,
        "optimizer": optim.__class__.__name__,
        "data_loss_weight": data_loss_weight,
        "mass_loss_weight": mass_loss_weight,
        "x_loss_weight": x_loss_weight,
        "y_loss_weight": y_loss_weight,
        "warmup_epochs": warmup_epochs,
    }
)

# Training / 네트워크 학습시키기
st_epoch = 0
net, optim, st_epoch, best_val, best_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) # 이전의 상태를 불러온다. 따라서 st_epoch가 달라지고, 현재 weight등을 전부 최신것으로 가져온다. 

delta = 1e-6  # 개선 판단 여유폭(원하면 1e-6 등으로) / 우리는 val_mean < best_val - delta 인 경우에만 model을 저장할 것이다. 

loss_log = {"epoch": [], "train": [], "val": [], "train_data_loss": [], "train_mass_loss": [], "train_x_loss": [], "train_y_loss": [], 
           "val_data_loss": [], "val_mass_loss": [], "val_x_loss": [], "val_y_loss": [], "best_epoch": [], "learning_rate": []}
if os.path.exists(my_log_file):
    with open(my_log_file, "rb") as f:
        loss_log = pickle.load(f)


for epoch in range(st_epoch+1, num_epoch+1):

    start_time = time.time()  # 🔹 epoch 시작 시간 기록
    
    net.train() # training이라는 것을 알려준다. / 실제로는 def에 train이라는 것이 없지만, nn.Module을 상속 받았기 때문에 가능한 것이다. nn.Module안에 이미 들어있고 // net.train()은 학습모드, net.eval()은 평가모드이다. 

    # ----------------------------
    # --- Dynamic Weight Averaging (Liu et al., 2019 CVPR) ---
    # ----------------------------
    if epoch >= 3 and num_tasks > 1:  # 직전 2 epoch 필요
        # ✅ 활성 task만 DWA 계산에 포함
        key_map = {
            "data": "train_data_loss",
            "mass": "train_mass_loss",
            "x":    "train_x_loss",
            "y":    "train_y_loss",
        }

        prev_loss = np.array([loss_log[key_map[t]][-1] for t in active_tasks], dtype=np.float64)
        prev2_loss = np.array([loss_log[key_map[t]][-2] for t in active_tasks], dtype=np.float64)

        w = prev_loss / (prev2_loss + 1e-8)
        z_overflow = w / T
        z_overflow = z_overflow - np.max(z_overflow) # 여기 부분에서 4개의 weights ratio에서 특정 값이 압도적으로 크다면 그 값을 전체적으로 빼준다 ; 그 다음에 exp를 하기 때문에 상대적 크기 관계는 그대로 유지된다.
        w = np.exp(z_overflow) # 여기서 종종 NaN이 터진다. 즉 w/T가 너무 커져서; 즉 T는 2고정이니까 w가 너무 커져서 exp가 inf로 터지는 사례가 발생함. 
        
        dwa_weights = num_tasks * w / (np.sum(w) + 1e-12)

        # 특정 loss 비중 강화 (data loss 강화는 data가 활성일 때만)
        boost_factor = 1
        if "data" in active_tasks:
            data_idx = active_tasks.index("data")
            dwa_weights[data_idx] *= boost_factor
            dwa_weights = num_tasks * dwa_weights / np.sum(dwa_weights)

        # ✅ 업데이트는 활성 task에만 반영, 비활성 task는 항상 0 유지
        new_w = dict(zip(active_tasks, dwa_weights.tolist()))
        data_loss_weight = float(new_w.get("data", 0.0))
        mass_loss_weight = float(new_w.get("mass", 0.0))
        x_loss_weight    = float(new_w.get("x",    0.0))
        y_loss_weight    = float(new_w.get("y",    0.0))

    # ----------------------------
    # ✅ Physics cap 적용
    # ----------------------------
    data_loss_weight = max(float(data_loss_weight), DATA_W_MIN)
    phys_sum = float(mass_loss_weight) + float(x_loss_weight) + float(y_loss_weight)
    phys_cap = PHYS_CAP_RATIO * float(data_loss_weight)
    if phys_sum > phys_cap and phys_sum > 0.0:
        _scale = phys_cap / phys_sum
        mass_loss_weight *= _scale
        x_loss_weight    *= _scale
        y_loss_weight    *= _scale

    # warm-up factor 계산 (0~1 사이 값이 나온다)
    # 초반 warmup_epochs 동안 physics loss를 점진적으로 증가시킴
    warmup_factor = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

    # 현재 epoch에서 사용할 weight (physics loss에만 warm-up 적용)
    mass_w = mass_loss_weight * warmup_factor
    x_w    = x_loss_weight * warmup_factor
    y_w    = y_loss_weight * warmup_factor

    run.log({
        "weights/c4.data": float(data_loss_weight),
        "weights/c3.mass": float(mass_w),
        "weights/c2.x": float(x_w),
        "weights/c1.y": float(y_w)
    }, step=epoch)
    
    loss_arr = []
    data_arr = []
    mass_arr = []
    x_arr = []
    y_arr = []
    loss_log["epoch"].append(epoch)

    current_lr = optim.param_groups[0]['lr']
    loss_log["learning_rate"].append(current_lr)                 # pkl에 저장 
    run.log({"my_system/d1.learning_rate": current_lr}, step=epoch)
    print(f"Epoch {epoch:04d} | Current LR: {current_lr:.6f} | Warmup factor: {warmup_factor:.4f}")
 
    for batch, data in enumerate(loader_train, 1): # batch는 순서를 말하는거고, data는 하나의 배치에 해당하는 데이터를 담고 있는 변수이다. 
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device) # 여기서의 dimension: torch.Size([4, 1, 512, 512]) # 즉 batch, channel, H, W

        #if batch > 2:
        #    break

        output = net(input) # 여기서 우리가 정의한 UNet이 training mode가 된 'net'이 input을 받아서 output을 출력한다. 

        # backward pass
        optim.zero_grad() # PyTorch는 기본적으로 gradient를 계속 누적하게 된다. 그래서 backward연산을 위해서 항상 gradient를 초기화 해주어야 한다.

        #data_loss = fn_loss(output[:, 0:1, :, :], label[:, 0:1, :, :]) # 모델의 출력인 ouptut과 label을 비교해서 loss를 계산하는 단계
        data_loss = fn_loss(output, label)

        mass_loss, x_loss, y_loss = compute_batch_momentum_loss(
                                                                input, output,
                                                                input_min_gpu, input_max_gpu,
                                                                output_min_gpu, output_max_gpu,
                                                                n_gpu=n_gpu
                                                            )

        Total_loss = data_loss_weight*data_loss + mass_w*mass_loss + x_w*x_loss + y_w*y_loss

        # --- 문제가 생기면 다시 실행 --- / 근데 이 문제가 생기면 절대 안된다. 이전에는 sqrt()로 문제가 발생. 하지만 이 문제는 해결함. 
        if torch.isnan(Total_loss):
            print(f"❌ NaN detected at Epoch {epoch}, Batch {batch}. Reloading checkpoint...")
            net, optim, st_epoch, best_val, best_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
            continue

        with torch.autograd.set_detect_anomaly(True):
            Total_loss.backward()

        #Total_loss.backward() # 여기서 gradient를 계산하게 된다. loss를 기준으로 작성하고, backpropagation이 되게 된다. 각 layer의 weight와 bias가 손실에 얼마나 기여했는지를 자동으로 계산해 주게 된다. 
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5) # gradient 폭발을 막는다. 1~5사이의 값이 많이 쓰이고, 1은 강하게 제안해서 안정적이지만 학습이 느려질 수 있음; 5는 느슨하게 제한해서 폭발만 막는 수준이다. 
        
        optim.step() # 여기서는 optimizer가 현재 계산된 gradient를 바탕으로 파라미터 업데이터를 수행하는 것이다. 

        # 손실함수 계산
        loss_arr += [Total_loss.item()] # loss.item()은 tensor를 float로 바꿔주고, 이번 배치의 손실 값을 저장하게 된다. 
        data_arr += [data_loss.item()]
        mass_arr += [mass_loss.item()]
        x_arr += [x_loss.item()]
        y_arr += [y_loss.item()]

        #print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
        #     (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr))) # 현재 상태, epoch, batch상태, 그리고 loss를 averaged해서 알려준다. 
        

    # loss 저장
    loss_log["train"].append(float(np.mean(loss_arr)))
    loss_log["train_data_loss"].append(float(np.mean(data_arr)))
    loss_log["train_mass_loss"].append(float(np.mean(mass_arr)))
    loss_log["train_x_loss"].append(float(np.mean(x_arr)))
    loss_log["train_y_loss"].append(float(np.mean(y_arr)))

    # Neptune 저장
    train_mean = float(np.mean(loss_arr))
    # ----------------------------
    # ✅ r 비율 출력 (epoch당 1회)
    # r = (wx*Lx + wy*Ly + wm*Lm) / (wd*Ld)
    # ----------------------------
    _Ld = float(np.mean(data_arr)) if len(data_arr) > 0 else 0.0
    _Lx = float(np.mean(x_arr)) if len(x_arr) > 0 else 0.0
    _Ly = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.0
    _Lm = float(np.mean(mass_arr)) if len(mass_arr) > 0 else 0.0
    _num = float(x_w) * _Lx + float(y_w) * _Ly + float(mass_w) * _Lm
    _den = float(data_loss_weight) * _Ld + 1e-12
    r_ratio = _num / _den
    print(f"[Physics/Data ratio r] epoch={epoch:04d} | r={r_ratio:.6f} | wd={data_loss_weight:.3f} | wx={x_w:.3f} wy={y_w:.3f} wm={mass_w:.3f}")

    run.log({"weights/r_ratio": float(r_ratio)}, step=epoch)

    run.log({
        "train/a5.loss": train_mean,
        "train/a4.data_loss": float(np.mean(data_arr)),
        "train/a3.mass_loss": float(np.mean(mass_arr)),
        "train/a2.x_loss": float(np.mean(x_arr)),
        "train/a1.y_loss": float(np.mean(y_arr)),
    }, step=epoch)
    
    if torch.cuda.is_available(): # GPU 사용량 저장 
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)  # 내 GPU가 실제로 얼마나 쓰이는지 GPU 전체 점유율; GB 단위/ 즉 Byte에 1024^3을 해서 GB값으로 바꿔준 것임. 
        gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3) # GPU전체 점유율 / 이거는 PyTorch가 확보한 전체 GPU영역이라 사용되지 않는 부분도 있다. / 이 값이 최대 값인 (현재 desktop의 최대값은 24GB임) 24에 도달하면 위험하다. 

        run.log({
            "my_system/d2.gpu_mem_alloc(GB)": gpu_mem_alloc,
            "my_system/d3.gpu_mem_reserved(GB)": gpu_mem_reserved,
        }, step=epoch)

    # --- Validation ---
    
    with torch.no_grad(): # with 는 context manager이다. 이 블록 동안은 ~~한 규칙을 적용하고, 끝나면 자동으로 정리해라 하는 의미. 이 경우에는 이 안에서 실행되는 연산에서 autograd를 추적하지 않도록 한다. // PyTorch는 기본적으로 모든 연산을기록한다. 왜냐면 나중에 loss.backward()를 할 때 자동으로 gradient를 계산하기 때문에. training단계에서는 weight의 업데이트를 위해서 필요하지만, validation에서는 단순히 output확인만 하면되기 때문에 이 autograd를 꺼주는 것이다. / (1) 메모리 절약 (2) 속도 향상 (3) 불필요한 gradient 계산 방지
        net.eval() # 모델을 평가 모드로 전환하는 것이다. 실제로 몇몇 레이어의 동작이 달라지는데 예를 들면: (1) 학습 할 때는 dropout처럼 몇몇 뉴런을 랜덤하게 꺼주지만, 평가할 때는 모든 뉴런을 켜둔다. (2) batchnorm 학습할 때는 배치 통계 (평균, 분산) 을 사용하지만, 평가할 때는 학습중에 축적된 running statistics를 사용한다. 즉 net.eval()을 해주지 않으면, validation할 때 결과가 들쭉날쭉 하기 때문에 reproducibility가 깨지게 된다. 
        loss_arr = []
        data_arr = []
        mass_arr = []
        x_arr = []
        y_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            #if batch > 2:
            #    break

            output = net(input)

            # 손실함수 계산하기
            #data_loss = fn_loss(output[:, 0:1, :, :], label[:, 0:1, :, :])
            data_loss = fn_loss(output, label)

            mass_loss, x_loss, y_loss = compute_batch_momentum_loss(
                                                                input, output,
                                                                input_min_gpu, input_max_gpu,
                                                                output_min_gpu, output_max_gpu,
                                                                n_gpu=n_gpu
                                                            )

            Total_loss = data_loss_weight*data_loss + mass_w*mass_loss + x_w*x_loss + y_w*y_loss
            
            loss_arr += [Total_loss.item()]
            data_arr += [data_loss.item()]
            mass_arr += [mass_loss.item()]
            x_arr += [x_loss.item()]
            y_arr += [y_loss.item()]

            #print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.8f" %
            # (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

    # loss 저장
    loss_log["val"].append(float(np.mean(loss_arr)))
    loss_log["val_data_loss"].append(float(np.mean(data_arr)))
    loss_log["val_mass_loss"].append(float(np.mean(mass_arr)))
    loss_log["val_x_loss"].append(float(np.mean(x_arr)))
    loss_log["val_y_loss"].append(float(np.mean(y_arr)))

    # W&B 저장
    val_mean = float(np.mean(loss_arr))
    run.log({
        "val/b5.loss": val_mean,
        "val/b4.data_loss": float(np.mean(data_arr)),
        "val/b3.mass_loss": float(np.mean(mass_arr)),
        "val/b2.x_loss": float(np.mean(x_arr)),
        "val/b1.y_loss": float(np.mean(y_arr)),
    }, step=epoch)
    
    scheduler.step(val_mean)  # val_mean은 validation loss 평균 / # 🔑 여기서 lr 업데이트
    
    # =====================================================
    # ✅ 15 epoch 후 초기 조건 체크 및 재시작
    # =====================================================
    if epoch == 3:
        # 4개의 loss 값 확인
        val_x_loss = float(np.mean(x_arr))
        val_y_loss = float(np.mean(y_arr))
        train_x_loss = loss_log["train_x_loss"][-1] if len(loss_log["train_x_loss"]) > 0 else 0.0
        train_y_loss = loss_log["train_y_loss"][-1] if len(loss_log["train_y_loss"]) > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"15 epoch 체크:")
        print(f"  val/b2.x_loss: {val_x_loss:.10f}")
        print(f"  val/b1.y_loss: {val_y_loss:.10f}")
        print(f"  train/a2.x_loss: {train_x_loss:.10f}")
        print(f"  train/a1.y_loss: {train_y_loss:.10f}")
        print(f"{'='*80}")
        
        # 조건 1: 0.00001보다 작은지 확인
        threshold = 0.00001
        too_small = (val_x_loss < threshold or val_y_loss < threshold or 
                    train_x_loss < threshold or train_y_loss < threshold)
        
        # 조건 2: 15 epoch까지 값이 일관성있게 같은지 확인
        if len(loss_log["val_x_loss"]) >= 15 and len(loss_log["val_y_loss"]) >= 15:
            val_x_all_same = len(set(loss_log["val_x_loss"][:15])) == 1
            val_y_all_same = len(set(loss_log["val_y_loss"][:15])) == 1
        else:
            val_x_all_same = False
            val_y_all_same = False
            
        if len(loss_log["train_x_loss"]) >= 15 and len(loss_log["train_y_loss"]) >= 15:
            train_x_all_same = len(set(loss_log["train_x_loss"][:15])) == 1
            train_y_all_same = len(set(loss_log["train_y_loss"][:15])) == 1
        else:
            train_x_all_same = False
            train_y_all_same = False
        
        all_same = val_x_all_same or val_y_all_same or train_x_all_same or train_y_all_same
        
        # 조건 만족 시 재시작
        if too_small or all_same:
            print(f"\n❌ 초기 조건 문제 감지!")
            if too_small:
                print(f"   → 일부 loss 값이 {threshold}보다 작습니다.")
            if all_same:
                print(f"   → 일부 loss 값이 15 epoch 동안 동일합니다.")
            
            # 재시작 횟수 확인 (최대 10번)
            max_restarts = 10
            if restart_count >= max_restarts:
                print(f"\n⚠️ 재시작 횟수 제한 도달 ({max_restarts}번)")
                print(f"   → 초기 조건 문제가 지속되어 스크립트를 종료합니다.")
                print(f"{'='*80}\n")
                
                # wandb 종료
                if 'run' in globals():
                    run.finish()
                
                # 스크립트 종료
                sys.exit(1)
            else:
                print(f"   → Seed를 재설정하고 처음부터 재시작합니다... (재시작 횟수: {restart_count + 1}/{max_restarts})")
                
                # 새로운 seed 생성 (0~100000) - random 모듈 대신 os.urandom 사용
                import struct
                import shutil
                new_seed = struct.unpack('I', os.urandom(4))[0] % 100001
                print(f"   → 새로운 seed: {new_seed}")
                
                # wandb 종료
                if 'run' in globals():
                    run.finish()
                
                # 기존 checkpoint 디렉토리 삭제
                if os.path.exists(ckpt_dir):
                    print(f"   → 기존 checkpoint 디렉토리 삭제: {ckpt_dir}")
                    shutil.rmtree(ckpt_dir)
                
                # 기존 log 파일 삭제
                if os.path.exists(my_log_file):
                    print(f"   → 기존 log 파일 삭제: {my_log_file}")
                    os.remove(my_log_file)
                
                # 임시 데이터 폴더 삭제
                if os.path.exists(data_dir):
                    print(f"   → 임시 데이터 폴더 삭제: {data_dir}")
                    shutil.rmtree(data_dir)
                
                # 스크립트 재실행
                # seed와 재시작 횟수를 환경변수로 전달
                os.environ['RESTART_SEED'] = str(new_seed)
                os.environ['RESTART_COUNT'] = str(restart_count + 1)
                print(f"   → 스크립트를 재시작합니다...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            print(f"✅ 초기 조건 정상 - 학습 계속 진행")
            print(f"{'='*80}\n")

    # 모든 epoch에서 save를 해주고, 만약 best가 나왔다면 그것의 이름을 best model로 해서 새로 지어주자. 
    if val_mean < best_val - delta:
        best_val = val_mean
        best_epoch = epoch
        bestsave(ckpt_dir=ckpt_dir, net=net, optim=optim) # 가장 좋은 상황일 때만 저장을 한다. 
        loss_log["best_epoch"].append(best_epoch)
        patience_counter = 0  # 개선되었으므로 patience 카운터 리셋
    else:
        patience_counter += 1  # 개선되지 않았으므로 카운터 증가

    # Early Stopping
    if patience_counter >= early_stop_patience:
        print(f"\n 📌 Early stopping triggered at epoch {epoch}! (no improvement for {early_stop_patience} epochs)")
        break
        
    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, best_val=best_val, best_epoch=best_epoch)
        
    with open(my_log_file, "wb") as f: # 내가 직접 저장하는 training, validation loss값 
        pickle.dump(loss_log, f)

    # Memory 관리 
    gc.collect() # python-level memory 해제
    del input, output, label  # 불필요한 variables 제거 
    torch.cuda.empty_cache() # GPU 메모리 캐시 해제/ 할당은 끝났지만, 아직 사용중이 아닌 tensor들이 GPU RAM을 차지하고 있을 때 제거해 주는 함수 

    # ----------------------
    # 🔹 epoch 시간 출력
    # ----------------------
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Epoch {epoch} completed in {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")

    # W&B 시간 저장
    run.log({"my_system/d4.epoch_time(sec)": elapsed}, step=epoch)




# In[ ]:





# In[ ]:





# In[ ]:


'''
Testing
'''

# 하이퍼 파라미터 설정
lr = globals().get("lr", 1e-3)
batch_size = globals().get("batch_size", 6)
num_epoch = globals().get("num_epoch", 100)

# 각 변수 torch로 변환
input_min_gpu  = torch.tensor(input_min,  device="cuda").view(1, -1, 1, 1)
input_max_gpu   = torch.tensor(input_max,   device="cuda").view(1, -1, 1, 1)
output_min_gpu = torch.tensor(output_min, device="cuda").view(1, -1, 1, 1)
output_max_gpu  = torch.tensor(output_max,  device="cuda").view(1, -1, 1, 1)

# GPU, CPU를 정해주는거
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 내 환경에서 cuda가 지원 가능하다면 device가 cuda (GPU)가 되는거고, 아니라면 cpu를 하라는 것이다. 

# 네트워크 학습하기 / 
transform = transforms.Compose([ToTensor()]) # 원래 부분에서 Random Flip 부분을 제거 

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0) # 데이터를 그냥 가져오는게 아니라 학습 과정에 맞게 효율적이게 가져온다./ 학습 시 데이터를 한번에 batch_size개씩 묶어서 모델에 주게 된다. num_workers는 CPU N개에서 병렬처리를 한다는 것이다. 

# 네트워크 생성
#net = UNet().to(device) # 내가 정의한 UNet 모델클래스의 객체 (인스턴스)를 생성하였다. 위에 device를 정의해 주었다. cpu또는 cuda에서 돌릴 준비를 한다는 것이다. 상황에 따라서 맞춘다. 

# loss function
fn_loss = nn.MSELoss().to(device) # 이거는 Binary Cross Entropy with Logits Loss 로써 이진분류로 픽셀 단위의 이진 segmentation에서 사용된다. 
#fn_loss = lambda yhat, y: torch.sqrt(nn.MSELoss()(yhat, y))

# Optimizer
optim = torch.optim.AdamW(net.parameters(), lr=lr)

# 부수적인 variables설정
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size) # 몇번 batch size가 돌아가야 하는지 알려준다. 

# 부수적인 function / # 간단한 함수를 만들 때 쓰는 문법이다. // lambda 매개변수 : 리턴값 구조
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # 여기서는 CPU로 가져와서, 학습 그래프에서 detach하고 (gradient 추적X), torch tensor를 numpy배열로 변환한 이후, batch, channel, height, width 의 torch 순서를 batch, height, width, channel 로 바꿔주는 과정이다. 
    # 즉 numpy 이미지 형태로 바꿔주는 기능을 한다. 
fn_denorm = lambda x, mean, std: (x * std) + mean # Normalization한 데이터를 원래의 스케일로 되돌리는 함수. 
fn_denorm2 = lambda x, min, max: (x * (max-min)) + min # MinMax scaling에서 Normalization한 데이터를 원래의 스케일로 되돌리는 함수. 
#fn_class = lambda x: 1 * (x>0.5)


# In[ ]:


result_dir = Main_dir + '/outputs/RESULTS_' + Unique_number
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# Testing
st_epoch = 0
net, optim, st_epoch, best_val, best_epoch = bestload(ckpt_dir=ckpt_dir, net=net, optim=optim) # Test set이기 때문에 Best 상태를 불러온다.

with torch.no_grad(): # with 는 context manager이다. 이 블록 동안은 ~~한 규칙을 적용하고, 끝나면 자동으로 정리해라 하는 의미. 이 경우에는 이 안에서 실행되는 연산에서 autograd를 추적하지 않도록 한다. // PyTorch는 기본적으로 모든 연산을기록한다. 왜냐면 나중에 loss.backward()를 할 때 자동으로 gradient를 계산하기 때문에. training단계에서는 weight의 업데이트를 위해서 필요하지만, validation에서는 단순히 output확인만 하면되기 때문에 이 autograd를 꺼주는 것이다. / (1) 메모리 절약 (2) 속도 향상 (3) 불필요한 gradient 계산 방지
    net.eval() # 모델을 평가 모드로 전환하는 것이다. 실제로 몇몇 레이어의 동작이 달라지는데 예를 들면: (1) 학습 할 때는 dropout처럼 몇몇 뉴런을 랜덤하게 꺼주지만, 평가할 때는 모든 뉴런을 켜둔다. (2) batchnorm 학습할 때는 배치 통계 (평균, 분산) 을 사용하지만, 평가할 때는 학습중에 축적된 running statistics를 사용한다. 즉 net.eval()을 해주지 않으면, validation할 때 결과가 들쭉날쭉 하기 때문에 reproducibility가 깨지게 된다. 
    loss_arr = []
    for batch, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = net(input)
        
        # 손실함수 계산하기
        data_loss = fn_loss(output, label)
        
        mass_loss, x_loss, y_loss = compute_batch_momentum_loss(
                                                                input, output,
                                                                input_min_gpu, input_max_gpu,
                                                                output_min_gpu, output_max_gpu,
                                                                n_gpu=n_gpu
                                                            )
        
        Total_loss = data_loss_weight*data_loss + mass_loss_weight*mass_loss + x_loss_weight*x_loss + y_loss_weight*y_loss
        
        loss_arr += [Total_loss.item()]
        #print("TEST: BATCH %04d / %04d | LOSS %.8f" %
        # (batch, num_batch_test, np.mean(loss_arr)))
        
        label = fn_tonumpy(fn_denorm2(label, min=output_min_gpu, max=output_max_gpu)) # Normalized되어있는 결과를 denormalize해주기 
        #input = fn_tonumpy(fn_denorm2(input, min=input_min_gpu, max=input_max_gpu))
        output = fn_tonumpy(fn_denorm2(output, min=output_min_gpu, max=output_max_gpu))

        for j in range(label.shape[0]):
            id = batch_size * (batch - 1) + j

            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())
            #np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
         (batch, num_batch_test, np.mean(loss_arr)))


# In[ ]:


import os
import numpy as np
from sklearn.metrics import f1_score

result_dir2 = os.path.join(result_dir, "numpy")

lst_data = os.listdir(result_dir2)
lst_label = sorted([f for f in lst_data if f.startswith('label')])
lst_output = sorted([f for f in lst_data if f.startswith('output')])

# ========================
# 함수 정의
# ========================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def compute_f1(label_h, output_h, threshold=0.05):
    """
    Flood extent F1-score 계산
    label_h: 2D array (ground truth water depth)
    output_h: 2D array (predicted water depth)
    threshold: flooding 여부를 나누는 수위 기준 (기본값 0.05 m)
    """
    label_mask = (label_h > threshold).astype(int)
    output_mask = (output_h > threshold).astype(int)
    return f1_score(label_mask.flatten(), output_mask.flatten(), zero_division=0)

# ========================
# 전체 RMSE / F1 계산
# ========================
rmse_all = []
f1_all = []

for fname_label, fname_output in zip(lst_label, lst_output):
    label = np.load(os.path.join(result_dir2, fname_label))
    output = np.load(os.path.join(result_dir2, fname_output))

    # --- RMSE (h 채널만)
    rmse_val = rmse(label[:, :, 0], output[:, :, 0])
    rmse_all.append(rmse_val)

    # --- Flood extent F1-score
    f1_val = compute_f1(label[:, :, 0], output[:, :, 0], threshold=0.02)
    f1_all.append(f1_val)

# ========================
# 결과 출력
# ========================
print(f"총 데이터 개수: {len(rmse_all)}")
print(f"h 전체 평균 RMSE: {np.mean(rmse_all):.6f}")
print(f"\n최소 RMSE: {np.min(rmse_all):.6f}")
print(f"최대 RMSE: {np.max(rmse_all):.6f}")

print("\n============================")
print(f"Flood extent 평균 F1-score: {np.mean(f1_all):.6f}")
print(f"최소 F1-score: {np.min(f1_all):.6f}")
print(f"최대 F1-score: {np.max(f1_all):.6f}")


# In[ ]:


run.config["Test_mean_RMSE"] = float(np.mean(rmse_all))
run.config["Test_min_RMSE"]  = float(np.min(rmse_all))
run.config["Test_max_RMSE"]  = float(np.max(rmse_all))

run.finish()


# In[ ]:


'''
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
d_result_maker.py의 결과 분석 로직 통합
이미 계산된 X_test, Y_test를 사용하여 상세한 메트릭 계산 및 저장

⚠️ 중요: 이 부분에서 사용하는 X_test, Y_test는 b_running_FDM.py의 266-269줄에서
이미 val_event_order와 test_event_order로 올바르게 분할된 데이터입니다.
- val_event_order, test_event_order는 원본 7개 이벤트 기준 (1~7 범위)
- 7개 이벤트 모두 사용하며, Ike(4번)는 test, Imelda(6번)는 validation으로 사용
'''

print("\n" + "="*80)
print("결과 분석 시작 (d_result_maker.py 로직 통합)")
print(f"✅ 사용 중인 val_event_order: {val_event_order} (Imelda), test_event_order: {test_event_order} (Ike) - 원본 7개 기준")
print("="*80)

# 결과 파일 로드 (numpy 파일에서)
result_dir2 = os.path.join(result_dir, "numpy")
lst_data = sorted(os.listdir(result_dir2))
lst_label = sorted([f for f in lst_data if f.startswith('label')])
lst_output = sorted([f for f in lst_data if f.startswith('output')])

def load_all_results_from_numpy(result_dir):
    """numpy 파일에서 결과를 로드하여 (596, 596, 3, N) 형태로 반환"""
    labels, outputs = [], []
    for fname_label, fname_output in zip(lst_label, lst_output):
        label = np.load(os.path.join(result_dir, fname_label))[:, :, :]
        output = np.load(os.path.join(result_dir, fname_output))[:, :, :]
        labels.append(label)
        outputs.append(output)
    
    # (596, 596, 3, N) 형태로 변환
    labels = np.stack(labels, axis=-1)
    outputs = np.stack(outputs, axis=-1)
    
    print(f"✅ Loaded {labels.shape[-1]} cases from {result_dir}")
    print(f"   Data shape: {labels.shape} (ny, nx, channels, N)")
    return labels, outputs

# numpy 파일에서 결과 로드
labels_np, outputs_np = load_all_results_from_numpy(result_dir2)

# ========================
# 1. 시간별 메트릭 계산
# ========================
def compute_timeseries_metrics(labels, outputs, threshold=0.02):
    """
    (ny, nx, N) 형태의 label, output 데이터를 입력받아
    각 time step별 RMSE, CSI, NSE, KGE 값을 계산.
    """
    ny, nx, N = labels.shape
    rmse_all = np.zeros(N)
    csi_all = np.zeros(N)
    nse_all = np.zeros(N)
    kge_all = np.zeros(N)

    for i in range(N):
        label_h = labels[:, :, i]
        output_h = outputs[:, :, i]

        # RMSE
        rmse_all[i] = np.sqrt(np.mean((label_h - output_h) ** 2))

        # CSI
        label_mask = (label_h > threshold).astype(int)
        output_mask = (output_h > threshold).astype(int)

        TP = np.sum((label_mask == 1) & (output_mask == 1))
        FP = np.sum((label_mask == 0) & (output_mask == 1))
        FN = np.sum((label_mask == 1) & (output_mask == 0))
        denom = TP + FP + FN
        csi_all[i] = TP / denom if denom > 0 else np.nan

        # NSE
        obs = label_h.flatten()
        sim = output_h.flatten()
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        nse_all[i] = 1 - (numerator / denominator) if denominator != 0 else np.nan

        # KGE 계산
        mu_obs = np.mean(obs)
        mu_sim = np.mean(sim)
        std_obs = np.std(obs)
        std_sim = np.std(sim)

        if std_obs == 0 or std_sim == 0:
            r = np.nan
        else:
            r = np.corrcoef(obs, sim)[0, 1]

        alpha = std_sim / std_obs if std_obs != 0 else np.nan
        beta = mu_sim / mu_obs if mu_obs != 0 else np.nan

        if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
            kge_all[i] = np.nan
        else:
            kge_all[i] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return rmse_all, csi_all, nse_all, kge_all

rmse_h_time, csi_h_time, nse_h_time, kge_h_time = compute_timeseries_metrics(labels_np[:,:,0,:], outputs_np[:,:,0,:], threshold=0.01)
rmse_u_time, _, nse_u_time, kge_u_time = compute_timeseries_metrics(labels_np[:,:,1,:], outputs_np[:,:,1,:], threshold=0.01)
rmse_v_time, _, nse_v_time, kge_v_time = compute_timeseries_metrics(labels_np[:,:,2,:], outputs_np[:,:,2,:], threshold=0.01)

# ========================
# 2. 2D 메트릭 계산
# ========================
def compute_nse_2d(outputs, labels, order_):
    sim = outputs[:, :, order_, :]
    obs = labels[:, :, order_, :]

    obs_mean = np.mean(obs, axis=2, keepdims=True)

    num = np.sum((sim - obs)**2, axis=2)
    den = np.sum((obs - obs_mean)**2, axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        nse_2d = 1 - (num / den)

    nse_2d[den == 0] = np.nan
    return nse_2d

rmse_h_2D = np.sqrt(np.mean((outputs_np[:,:,0,:] - labels_np[:,:,0,:]) ** 2, axis=2))
rmse_u_2D = np.sqrt(np.mean((outputs_np[:,:,1,:] - labels_np[:,:,1,:]) ** 2, axis=2))
rmse_v_2D = np.sqrt(np.mean((outputs_np[:,:,2,:] - labels_np[:,:,2,:]) ** 2, axis=2))
nse_h_2D = compute_nse_2d(outputs_np, labels_np, 0)
nse_u_2D = compute_nse_2d(outputs_np, labels_np, 1)
nse_v_2D = compute_nse_2d(outputs_np, labels_np, 2)

# ========================
# 3. Peak-based window 계산
# ========================
mean_h_time_series = np.mean(labels_np[:,:,0,:], axis=(0,1))
peak_idx = np.argmax(mean_h_time_series)
peak_value = mean_h_time_series[peak_idx]

window = 12
start_idx = max(0, peak_idx - window)
end_idx   = min(len(mean_h_time_series), peak_idx + window + 1)

window_indices = np.arange(start_idx, end_idx)
mean_RMSE_h_window = np.mean(rmse_h_time[window_indices])
print('Total mean RMSE of h: ', np.mean(rmse_h_time))
print('Mean RMSE of h in peak-based window: ', mean_RMSE_h_window)

# ========================
# 4. Physics loss 계산
# ========================
# Global parameters
dt = 3600
theta = 35  # 단위는 도
dxi = 200
deta = 200
g = 9.81  # m/s^2
rho = 1024  # 물의 밀도
rho_a = 1.25  # 공기 밀도 [kg/m^3]

def crop_to_594x594(arr: np.ndarray) -> np.ndarray:
    """NumPy array 전용. (596,596) → (594,594)"""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    
    h, w = arr.shape[-2], arr.shape[-1]
    
    if h == 596:
        arr = arr[1:-1, :]
    elif h != 594:
        raise ValueError(f"Unexpected height: {h}")
    
    if w == 596:
        arr = arr[:, 1:-1]
    elif w != 594:
        raise ValueError(f"Unexpected width: {w}")
    
    return arr

def FDM_physics_loss(output_data, input_data, center_idx):
    """FDM physics loss 계산"""
    vars_dict = {
        "t-1": {"h": output_data[:,:,0,center_idx-1], "u": output_data[:,:,1,center_idx-1], "v": output_data[:,:,2,center_idx-1]},
        "t"  : {"h": output_data[:,:,0,center_idx], "u": output_data[:,:,1,center_idx], "v": output_data[:,:,2,center_idx]},
        "t+1": {"h": output_data[:,:,0,center_idx+1], "u": output_data[:,:,1,center_idx+1], "v": output_data[:,:,2,center_idx+1]},
    }
    h_tm1, u_tm1, v_tm1 = vars_dict["t-1"]["h"], vars_dict["t-1"]["u"], vars_dict["t-1"]["v"]
    h_t,   u_t,   v_t   = vars_dict["t"]["h"],   vars_dict["t"]["u"],   vars_dict["t"]["v"]
    h_tp1, u_tp1, v_tp1 = vars_dict["t+1"]["h"], vars_dict["t+1"]["u"], vars_dict["t+1"]["v"]
    
    # 상수 추출 (input_data shape: (ny, nx, channels, N))
    # B와 n은 시간에 따라 변하지 않는 상수이므로 첫 번째 timestep (center_idx=0) 사용
    B = input_data[:,:,17,0]  # bathymetry (채널 17, 첫 timestep)
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    U10x = input_data[:,:,13,center_idx]  # wind_u at center_idx
    U10y = input_data[:,:,14,center_idx]  # wind_v at center_idx
    n = input_data[:,:,18,0]  # manning_n (채널 18, 첫 timestep)
    
    # 1) dhu/dt & dv/dt
    dhu_dt = crop_to_594x594(((h_tp1 * u_tp1) - (h_tm1 * u_tm1)) / (2 * dt))
    dhv_dt = crop_to_594x594(((h_tp1 * v_tp1) - (h_tm1 * v_tm1)) / (2 * dt))
    
    # 2) d(hu^2)/dx & d(hv^2)/dy
    dhu2_dxi = ((h_t*u_t*u_t)[:,2:] - (h_t*u_t*u_t)[:,:-2]) / (2 * dxi)
    dhu2_deta = ((h_t*u_t*u_t)[2:,:] - (h_t*u_t*u_t)[:-2,:]) / (2 * deta)
    dhu2_dx = cos_theta*crop_to_594x594(dhu2_dxi) - sin_theta*crop_to_594x594(dhu2_deta)
    dhv2_dxi = ((h_t*v_t*v_t)[:,2:] - (h_t*v_t*v_t)[:,:-2]) / (2 * dxi)
    dhv2_deta = ((h_t*v_t*v_t)[2:,:] - (h_t*v_t*v_t)[:-2,:]) / (2 * deta)
    dhv2_dy = sin_theta*crop_to_594x594(dhv2_dxi) + cos_theta*crop_to_594x594(dhv2_deta)
    
    # 3) d(hvu)/dy & d(huv)/dx
    dhvu_dxi = ((h_t*u_t*v_t)[:,2:] - (h_t*u_t*v_t)[:,:-2]) / (2 * dxi)
    dhvu_deta = ((h_t*u_t*v_t)[2:,:] - (h_t*u_t*v_t)[:-2,:]) / (2 * deta)
    dhuv_dy = sin_theta*crop_to_594x594(dhvu_dxi) + cos_theta*crop_to_594x594(dhvu_deta)
    dhuv_dx = cos_theta*crop_to_594x594(dhvu_dxi) - sin_theta*crop_to_594x594(dhvu_deta)
    
    # 4) gh*d(h+B)/dx
    hB = h_t + B
    up, down, left, right = h_t[:-2,1:-1], h_t[2:,1:-1], h_t[1:-1,:-2], h_t[1:-1,2:]
    h_threshold = 0.01
    mask_freesurface = ((up > h_threshold) & (down > h_threshold) & (left > h_threshold) & (right > h_threshold)).astype(int)
    dhB_dxi  = np.clip((hB[:, 2:] - hB[:, :-2]) / (2 * dxi),  -10, 10)
    dhB_deta = np.clip((hB[2:, :] - hB[:-2, :]) / (2 * deta), -10, 10)
    ghdhB_dx = g * crop_to_594x594(h_t) * ( cos_theta*crop_to_594x594(dhB_dxi) - sin_theta*crop_to_594x594(dhB_deta) ) * mask_freesurface
    ghdhB_dy = g * crop_to_594x594(h_t) * ( sin_theta*crop_to_594x594(dhB_dxi) + cos_theta*crop_to_594x594(dhB_deta) ) * mask_freesurface
    
    # 5) tau_b/ρ
    hmin, hdry = 1e-4, 1e-3
    mask = (h_t >= hdry).astype(u_t.dtype)
    speed = np.sqrt(u_t * u_t + v_t * v_t + 1e-3)
    h13 = (h_t + 1e-3) ** (1/3)
    Cf = g * (n**2) / h13
    taubx_rho = crop_to_594x594(Cf * speed * u_t * mask)
    tauby_rho = crop_to_594x594(Cf * speed * v_t * mask)
    
    # 6) tau_w/ρ
    U10_mag = np.sqrt(U10x**2 + U10y**2 + 1e-3)
    Cd = np.where(
            U10_mag < 28.0, 
            0.001,
            np.where(U10_mag < 50.0, 0.0025, 0.0015))
    tauwx_rho = crop_to_594x594((rho_a / rho) * Cd * U10_mag * U10x)
    tauwy_rho = crop_to_594x594((rho_a / rho) * Cd * U10_mag * U10y)
    
    # Residuals
    x_physics_loss = np.abs(dhu_dt + dhu2_dx + dhuv_dy + ghdhB_dx + taubx_rho - tauwx_rho)
    y_physics_loss = np.abs(dhv_dt + dhuv_dx + dhv2_dy + ghdhB_dy + tauby_rho - tauwy_rho)
    
    # Mass conservation equation
    dh_dt = crop_to_594x594((h_tp1 - h_tm1) / (2 * dt))
    
    Hu = (B + h_t)*u_t
    dHu_dxi  = (Hu[:, 2:] - Hu[:, :-2]) / (2 * dxi)
    dHu_deta = (Hu[2:, :] - Hu[:-2, :]) / (2 * deta)
    dHu_dx = cos_theta * crop_to_594x594(dHu_dxi) - sin_theta * crop_to_594x594(dHu_deta) * mask_freesurface
    
    Hv = (B + h_t)*v_t
    dHv_dxi  = (Hv[:, 2:] - Hv[:, :-2]) / (2 * dxi)
    dHv_deta = (Hv[2:, :] - Hv[:-2, :]) / (2 * deta)
    dHv_dy = sin_theta * crop_to_594x594(dHv_dxi) + cos_theta * crop_to_594x594(dHv_deta) * mask_freesurface
    
    # Source term
    precipitation = input_data[:,:,12,center_idx] / 1000 / 3600         # [mm/hr → m/s]
    discharge     = input_data[:,:,16,center_idx] / (dxi * deta) / 3600 # [m³/hr → m/s]
    S_mn = crop_to_594x594(precipitation + discharge)
    
    mass_physics_loss = np.maximum(dh_dt + dHu_dx + dHv_dy - S_mn, 0)

    return x_physics_loss, y_physics_loss, mass_physics_loss

def compute_phys_stats(output_data_596, input_data_596, eps=1e-12):
    """모든 timestep의 physics loss를 계산"""
    T = int(output_data_596.shape[3])
    
    x_list, y_list, mass_list = [], [], []
    
    for center_idx in range(1, T - 1):
        x_loss, y_loss, mass_loss = FDM_physics_loss(output_data_596, input_data_596, center_idx)
        x_list.append(x_loss)
        y_list.append(y_loss)
        mass_list.append(mass_loss)
        if (center_idx + 1) % 10 == 0:
            print(f"  Physics loss 계산 중: {center_idx+1}/{T-1}")
    
    x_all = np.stack(x_list, axis=-1)
    y_all = np.stack(y_list, axis=-1)
    mass_all = np.stack(mass_list, axis=-1)
    
    del x_list, y_list, mass_list
    
    x_time = np.mean(x_all, axis=(0, 1))
    y_time = np.mean(y_all, axis=(0, 1))
    mass_time = np.mean(mass_all, axis=(0, 1))
    
    x_2D = np.mean(x_all, axis=(2))
    y_2D = np.mean(y_all, axis=(2))
    mass_2D = np.mean(mass_all, axis=(2))

    return {
        'x_time': x_time,
        'y_time': y_time,
        'mass_time': mass_time,
        'x_2D': x_2D,
        'y_2D': y_2D,
        'mass_2D': mass_2D,
        'T': T,
        'N': T - 2,
    }

# Shape 변환: (N,596,596,3) -> (596,596,3,N)
labels_596 = np.transpose(Y_test, (1, 2, 3, 0))
outputs_596 = outputs_np  # 이미 (596,596,3,N) 형태

# X_test도 변환: (N,596,596,channels) -> (596,596,channels,N)
X_test_596 = np.transpose(X_test, (1, 2, 3, 0))

print(f"\noutputs shape: {outputs_596.shape}")
print(f"labels  shape: {labels_596.shape}")
print(f"X_test  shape: {X_test_596.shape}")

# Physics loss 계산
print("\n=== Physics loss 계산 시작 ===")
phys_pred = compute_phys_stats(outputs_596, X_test_596)
phys_label = compute_phys_stats(labels_596, X_test_596)

# Normalized physics loss
_eps = 1e-12

x_time = phys_pred['x_time']
y_time = phys_pred['y_time']
mass_time = phys_pred['mass_time']
x_time_label = phys_label['x_time']
y_time_label = phys_label['y_time']
mass_time_label = phys_label['mass_time']

x_2D = phys_pred['x_2D']
y_2D = phys_pred['y_2D']
mass_2D = phys_pred['mass_2D']

ratio_x_time = phys_pred['x_time'] / (phys_label['x_time'] + _eps)
ratio_y_time = phys_pred['y_time'] / (phys_label['y_time'] + _eps)
ratio_mass_time = phys_pred['mass_time'] / (phys_label['mass_time'] + _eps)

mean_ratio_x_time = np.mean(phys_pred['x_time']) / (np.mean(phys_label['x_time']) + _eps)
mean_ratio_y_time = np.mean(phys_pred['y_time']) / (np.mean(phys_label['y_time']) + _eps)
mean_ratio_mass_time = np.mean(phys_pred['mass_time']) / (np.mean(phys_label['mass_time']) + _eps)

ratio_x_2D = phys_pred['x_2D'] / (phys_label['x_2D'] + _eps)
ratio_y_2D = phys_pred['y_2D'] / (phys_label['y_2D'] + _eps)
ratio_mass_2D = phys_pred['mass_2D'] / (phys_label['mass_2D'] + _eps)

print('\n=== Normalized physics loss (pred/label) summary ===')
print('x_time   ratio mean:', mean_ratio_x_time)
print('y_time   ratio mean:', mean_ratio_y_time)
print('mass_time ratio mean:', mean_ratio_mass_time)

# ========================
# 5. 결과 저장
# ========================
save_file = os.path.join(Main_dir, 'outputs', '!Saved_results', f'z_RESULTS_{Unique_number}.npz')
os.makedirs(os.path.dirname(save_file), exist_ok=True)

np.savez_compressed(
    save_file,
    rmse_h_time=rmse_h_time,
    csi_h_time=csi_h_time,
    nse_h_time=nse_h_time,
    kge_h_time=kge_h_time,
    rmse_u_time=rmse_u_time,
    nse_u_time=nse_u_time,
    kge_u_time=kge_u_time,
    rmse_v_time=rmse_v_time,
    nse_v_time=nse_v_time,
    kge_v_time=kge_v_time,

    rmse_h_2D=rmse_h_2D,
    rmse_u_2D=rmse_u_2D,
    rmse_v_2D=rmse_v_2D,
    nse_h_2D=nse_h_2D,
    nse_u_2D=nse_u_2D,
    nse_v_2D=nse_v_2D,

    x_time=x_time,
    y_time=y_time,
    mass_time=mass_time,
    x_time_label=x_time_label,
    y_time_label=y_time_label,
    mass_time_label=mass_time_label,
    x_2D=x_2D,
    y_2D=y_2D,
    mass_2D=mass_2D,

    ratio_x_time=ratio_x_time,
    ratio_y_time=ratio_y_time,
    ratio_mass_time=ratio_mass_time,
    ratio_x_2D=ratio_x_2D,
    ratio_y_2D=ratio_y_2D,
    ratio_mass_2D=ratio_mass_2D,
    mean_ratio_x_time=mean_ratio_x_time,
    mean_ratio_y_time=mean_ratio_y_time,
    mean_ratio_mass_time=mean_ratio_mass_time,

    window_indices=window_indices,
    mean_RMSE_h_window=mean_RMSE_h_window
)

print(f"\n✅ 결과 저장 완료: {save_file}")
print("="*80)


# In[ ]:


import shutil
shutil.rmtree(data_dir)
print(f"임시 데이터 폴더 삭제 완료: {data_dir}")

