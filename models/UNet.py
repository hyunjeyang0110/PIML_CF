import os 
import numpy as np

import torch
import torch.nn as nn

class UNet(nn.Module): # UNet은 nn.Module을 상속 받았다. nn.Module은 PyTorch에서 모든 신경망 모델의 부모 클래스 (PyTorch에서 모델을 만들 때 모든 모델이 따라야 하는 기본 규칙이다). 여기서는 forward, backward, .parameters()같은 훈련에 필요한 기능이 이미 다 정의되어있다. 
    def __init__(self): # __init__는 생성자 (constructor)이다. 객체를 만들 때 가장 먼저 자동으로 실생되는 함수이다. 
        super(UNet, self).__init__() # 부모 클래스 (nn.Module)의 생성자(nn.Module.__init__)를 먼저 실행하라는 뜻이다.
            # 조금 햇갈리지만 흐름은 다음과 같다
        
            # model = UNet() 실행
            # UNet.__init__ 실행 시작
            # super(UNet, self).__init__() 호출 → nn.Module.__init__() 실행 (즉, 부모 클래스 초기화 먼저 함)
            # 그 다음에 self.enc1_1, self.pool1 같은 UNet만의 레이어 정의

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):   # Convolution, Batch normalization, ReLU, 2-D
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
    
            cbr = nn.Sequential(*layers) # 파란색 화살표인, convolution, batch, relu인 3개인 layer를 function으로 구현해서 함수로 만든 것이다. 
    
            return cbr

        # Contracting path (Encoder part)
        #elf.enc1_1 = CBR2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True) # 인코더 첫번째 스테이지 에서 첫번째 화살표
        self.enc1_1 = CBR2d(in_channels=19, out_channels=64) # kernel_size=3, stride=1, padding=1, bias=True 부분이 계속 동일하고, 이미 CBR2d definition으로 pre-define되어있기 때문에 이 부분을 제거하고 적어주어도 된다. 
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
    
        # 첫번째 max pooling 빨간색 화살표
        self.pool1 = nn.MaxPool2d(kernel_size=2)
    
        # Encoder 두번째 cycle
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
    
        # Encoder 세번째 cycle
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
    
        # Encoder 네번째 cycle
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
    
        # encoder 마지막 part
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
    
    # ================================================== 여기로부터 거의 대칭 ==================================================
        
        # Expansive path (Decoder part)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
    
        # 녹색 up_conv
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, # unpooling을 하는 것이지만, 그림의 크기가 늘어나는거지, channel의 숫자는 동일하다. 
                                         kernel_size=2, stride=2, padding=0, bias=True) # 이 part는 위에 self.pool4와 완전히 matching되는 부분이다. 
    
        # Decoder 네번째 cycle
        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512) # 여기서 이전의 상황을 가져오기 때문에 "copy and crop" 512가 아니라 1024가 되는 것이다. 
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        
        # Decoder 세번째 cycle
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                         kernel_size=2, stride=2, padding=0, output_padding=1, bias=True)
        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
    
        # Decoder 두번째 cycle
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                         kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
    
        # Decoder 첫번째 cycle
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                         kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CBR2d(in_channels=2*64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
    
        # 녹색 화살표 1*1 conv layer (prediction head)
        self.fc = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True) # padding은 가장자리를 늘려주는 작업이다. 근데 kernel_size가 1인 경우에는 입력과 출력의 2D 크기가 같게 된다. , 만약 kernel_size가 3이면 출력크기가 2가 줄어들기 때문에 padding을 1 해줘야 함. 즉 kernel_size가 1이면 padding을 0으로 해줘야 한다. 

    # 이 layer들을 연결해 보자. 
    def forward(self, x): # 여기서의 x는 input image이다. 
        
        # Encoder
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1) # 이 두 줄이 첫번째 encoder의 파란색 화살표이다. 
        pool1 = self.pool1(enc1_2) # 첫번째 encoder에 빨간색 화살표

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # Decoder
        dec5_1 = self.dec5_1(enc5_1)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # 여기가 "copy and crop" 부분이다. # dim 0:batch 1:channel, 2:height, 3:width // 여기서는 채널이 더해지는거니까 dim=1이다. 
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.fc(dec1_1)

        # h>=0 restriction
        x = torch.cat([torch.relu(x[:, 0:1, :, :]), x[:, 1:, :, :]], dim=1) # 첫번째 channel (h)는 relu를 적용해서 무조건 0 이상의 값으로 만들어 준다. 
        
        return x
