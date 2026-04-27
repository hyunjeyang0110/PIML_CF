import os 
import numpy as np

import torch
import torch.nn as nn


# Data loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir # 클래스 안에서는 __init__가 끝나면 지역 변수들은 모두 사라진다. 따라서 self.data_dir라는 인스턴스 변수로 저장해 주어야 한다. 
        self.transform = transform

        lst_data = os.listdir(self.data_dir)
        
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]
        
        lst_label.sort()
        lst_input.sort()
        
        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
    
        # 0에서 1로 normalize
        #label = label/255.0
        #input = input/255.0
    
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
    
        data = {'input': input, 'label': label}
    
        if self.transform:
            data = self.transform(data) # transform 함수가 정의되어있다면 바로 넣어준다는 뜻. / 이후에 내가 transform을 정의해 주고 활용할 것이다. 
    
        return data


# numpy -> tensor
class ToTensor(object):
    def __call__(self, data): # 여기서 data = {'input': input, 'label': label} 를 트랜스폼 하는걸 구현해 보자. 
        label, input = data['label'], data['input']

        label = label.transpose((2,0,1)).astype(np.float32) # 실제 numpy는 ,y,x,channel의 순서이지만, tensor는 channel,y,x의 순서이기 때문에 마지막 dimension을 가장 처음으로 옮겨 주어야 한다. 
        input = input.transpose((2,0,1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)} # 이게 numpy 배열에서 torch.Tensor로 변환해 주는 구간이다. 

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5): # 이미지를 0~1 범위로 이미 되어있는 경우에 x-0.5/0.5를 해주면 결과가 [-1, 1]로 매핑이 된다. 이거는 CNN, UNet같은 모델에서 학습 안정성을 높히기 위해서 많이 사용이 된다. 
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        data = {'label': label, 'input':input}
        
        return data

class RandomFlip(object): # 랜덤으로 좌우, 상하 반전 해주는 것. / 이건 Data augmentation을 위한 단계로써 데이터의 다양성을 만들어주고, 데이터를 여러 방식으로 변경해서 다양한 상황에서 견딜 수 있도록 만들어주는 것이다. 
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data