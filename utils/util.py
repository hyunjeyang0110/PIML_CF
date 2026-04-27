import os 
import gc
import glob
import numpy as np
import torch
import torch.nn as nn


def safe_state_dict(model):
    """
    GPU → CPU로 state_dict를 옮기고 tensor 참조를 해제해주는 함수.
    """
    return {k: v.cpu() for k, v in model.state_dict().items()}

# 네트워크 저장하기 
def save(ckpt_dir, net, optim, epoch, best_val, best_epoch, keep_last=2): # 이거는 PyTorch모델과 옵티마이저 상태를 저장하는 함수이다. 
    """
    매 epoch마다 checkpoint를 저장하되, 
    최근 keep_last개만 유지하고, 이전 것은 자동 삭제
    """
    if not os.path.exists(ckpt_dir):  # ckpt_dir이라는 directory가 없으면 새로 만들어준다. 
        os.makedirs(ckpt_dir)

    # GPU→CPU 변환
    net_cpu = safe_state_dict(net)
    optim_cpu = optim.state_dict()  # optim은 이미 CPU 기반이라 그대로 가능

    # 파일 이름 정의
    ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth") # 파일 이름의 규칙은, ckpt_dir/model_epochX.pth 의 형태가 된다. 

    torch.save(
        {
            'net': net_cpu,                # 네트워크 파라미터 # 네트워크의 파라미터(weight, bias etc.)를 딕셔너리 형태로 추출한다. / optimizer의 상태(학습률, momentum, 현재까지 계산된 모멘텀 값 등)을 저장해 준다. 이 두가리를 묶어서 저장해 주는 것이다. 
            'optim': optim_cpu,            # 옵티마이저 상태
            'epoch': epoch,                # 현재 epoch
            'best_val': best_val,          # 지금까지의 best validation loss
            'best_epoch': best_epoch       # best 모델이 나온 epoch
        },
        ckpt_path
    )

    # ✅ 오래된 checkpoint 자동 삭제
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "model_epoch*.pth")),
        key=os.path.getmtime
    )

    if len(ckpt_files) > keep_last:
        to_delete = ckpt_files[:-keep_last]
        for f in to_delete:
            try:
                os.remove(f)
                #print(f"Deleted old checkpoint: {f}")
            except Exception as e:
                print(f"⚠️ Failed to delete {f}: {e}")

    # 메모리 즉시 헤제
    del net_cpu, optim_cpu
    gc.collect()
    torch.cuda.empty_cache()


# Fine-tuning용 네트워크 저장하기 (자동 삭제 없음)
def finetuning_save(ckpt_dir, net, optim, epoch, best_val, best_epoch):
    """
    Fine-tuning 시 checkpoint를 저장하되, 자동 삭제하지 않음
    모든 checkpoint를 보존하여 fine-tuning 과정을 추적
    """
    if not os.path.exists(ckpt_dir):  # ckpt_dir이라는 directory가 없으면 새로 만들어준다. 
        os.makedirs(ckpt_dir)

    # GPU→CPU 변환
    net_cpu = safe_state_dict(net)
    optim_cpu = optim.state_dict()  # optim은 이미 CPU 기반이라 그대로 가능

    # 파일 이름 정의
    ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth") # 파일 이름의 규칙은, ckpt_dir/model_epochX.pth 의 형태가 된다. 

    torch.save(
        {
            'net': net_cpu,                # 네트워크 파라미터
            'optim': optim_cpu,            # 옵티마이저 상태
            'epoch': epoch,                # 현재 epoch
            'best_val': best_val,          # 지금까지의 best validation loss
            'best_epoch': best_epoch       # best 모델이 나온 epoch
        },
        ckpt_path
    )

    # ✅ Fine-tuning 모드에서는 자동 삭제하지 않음 (모든 checkpoint 보존)

    # 메모리 즉시 헤제
    del net_cpu, optim_cpu
    gc.collect()
    torch.cuda.empty_cache()


def bestsave(ckpt_dir, net, optim): # 이거는 PyTorch모델과 옵티마이저 상태를 저장하는 함수이다. 
    if not os.path.exists(ckpt_dir): # ckpt_dir이라는 directory가 없으면 새로 만들어준다. 
        os.makedirs(ckpt_dir)

    # GPU→CPU 변환
    net_cpu = safe_state_dict(net)
    optim_cpu = optim.state_dict()  # optim은 이미 CPU 기반이라 그대로 가능

    #torch.save({'net': net_cpu, 'optim': optim_cpu}, # 네트워크의 파라미터(weight, bias etc.)를 딕셔너리 형태로 추출한다. / optimizer의 상태(학습률, momentum, 현재까지 계산된 모멘텀 값 등)을 저장해 준다. 이 두가리를 묶어서 저장해 주는 것이다. 
    #          "./%s/model_BEST.pth" % (ckpt_dir))

    torch.save({'net': net_cpu, 'optim': optim_cpu},
           os.path.join(ckpt_dir, "model_BEST.pth"))


    del net_cpu, optim_cpu
    gc.collect()
    torch.cuda.empty_cache()    


# 네트워크 불러오기
#def load(ckpt_dir, net, optim): # 저장된 체크포인트를 불러와서 학습을 이어가기 위한 함수이다. 
#    if not os.path.exists(ckpt_dir):
#        epoch = 0 # 체크포인트 폴더가 아얘 없는 경우는 저장된 모델이 없으니까 처음부터 학습을 시작해야 한다. 즉 epoch = 0으로 초기화 해서 학습을 새로 시작해야 한다. 
#        return net, optim, epoch
#
#    ckpt_lst = os.listdir(ckpt_dir) # 해당 폴더에 있는 파일 목록을 가져온다.
#    ckpt_lst = [f for f in ckpt_lst if any(ch.isdigit() for ch in f)] # model_BEST.pth와 같이 숫자를 포함하고 있지 않은 것을 제외한다. 
#    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # 파일 이름에 들어있는 숫자를 기준으로 정렬한다. ex. ["model_epoch5.pth", "model_epoch10.pth"] → 정렬 후 맨 마지막 파일이 가장 최신 epoch.
#
#    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]), weights_only=True) # 가장 마지막, 가장 최신의 epoch모델을 불러온다. / torch.load는 저장된 딕셔너리를 다시 메모리로 불러오는 역할이다. 
#        # dict_model은 대부분 Network의 파라미터와 optimizer의 상태를 가지고 있고, dictionary형태이다. 위의 save부분을 보면 알 수 있음. 
#    net.load_state_dict(dict_model['net']) # 여기서 네트워크의 파라미터를 불러온다. 
#    optim.load_state_dict(dict_model['optim']) # 옵티마이저의 상태를 불러온다. 
#    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0]) # 파일 이름에서 epoch 번호만 추출. ex. "model_epoch10.pth" → "10" → 정수형 10.
#
#    return net, optim, epoch # 즉 불러온 네트워크, 옵티마이저, 마지막 학습 epoch번호를 함께 반환하게 된다. 이 값을 받아서 epoch+1부터 이어갈 수 있다. 


# 네트워크 불러오기
def load(ckpt_dir, net, optim): 
    if not os.path.exists(ckpt_dir):
        epoch = 0
        best_val = float('inf')
        best_epoch = -1
        return net, optim, epoch, best_val, best_epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if any(ch.isdigit() for ch in f)]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) 

    if len(ckpt_lst) == 0:  # ✅ 폴더가 비어있을 경우 추가 확인
        epoch = 0
        best_val = float('inf')
        best_epoch = -1
        return net, optim, epoch, best_val, best_epoch    

    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]), weights_only=False)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0]) # 파일 이름에서 epoch 번호만 추출. ex. "model_epoch10.pth" → "10" → 정수형 10.

    # best_val, best_epoch 추가
    best_val = dict_model.get('best_val', float('inf'))
    best_epoch = dict_model.get('best_epoch', -1)

    return net, optim, epoch, best_val, best_epoch

# 네트워크 BEST 불러오기
def bestload(ckpt_dir, net, optim):
    best_path = os.path.join(ckpt_dir, "model_BEST.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"BEST model not found at {best_path}")

    dict_model = torch.load(best_path, map_location="cpu", weights_only=False)  # BEST 모델 불러오기

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    # bestsave()에서는 epoch, best_val, best_epoch를 저장하지 않으니까 기본값 반환
    epoch = dict_model.get('epoch', -1)
    best_val = dict_model.get('best_val', float('inf'))
    best_epoch = dict_model.get('best_epoch', -1)

    return net, optim, epoch, best_val, best_epoch
