import torch
import torch.nn as nn
import torch.nn.init as init
from pytorch_ConvLSTM import ConvLSTMCell

class rdcnn_2_larger_angran(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_larger_angran, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(4, 84, 3, stride=2, padding=1),  # b, 84, 11, 11
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.Conv2d(84, 168, 3, stride=2, padding=1),  # b, 168, 6, 6
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),  # b, 168, 5, 5
            nn.Conv2d(168, 336, 3, stride=2, padding=1),  # b, 336, 3, 3
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),  # b, 336, 2, 2
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 3, stride=2, padding=1),  # b, 672, 3, 3
            nn.BatchNorm2d(672),
            nn.ReLU(True),
            nn.ConvTranspose2d(672, 336, 2, stride=2),  # b, 336, 6, 6
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.ConvTranspose2d(336, 84, 2, stride=2),  # b, 84, 12, 12
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.ConvTranspose2d(84, 1, 3, stride=2,padding=2),  # b, 1, 21, 21
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_larger(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_larger, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(4, 84, 3, stride=2,padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(84, 168, 3, stride=2, padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(168, 336, 3, stride=2, padding=1),
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 3, stride=2, padding=0),
            nn.BatchNorm2d(672),
            nn.ReLU(True),

            nn.ConvTranspose2d(672, 336, 3, stride=2, padding=0),
            nn.BatchNorm2d(336),
            nn.ReLU(True),

            nn.ConvTranspose2d(336, 84, 2, stride=1, padding=0),
            nn.BatchNorm2d(84),
            nn.ReLU(True),

            nn.ConvTranspose2d(84, 1, 2, stride=2,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_largerKernel(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_largerKernel, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(4, 84, 5, stride=2,padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(84, 168, 5, stride=2, padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(168, 336, 5, stride=2, padding=1),
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 5, stride=2, padding=0),
            nn.BatchNorm2d(672),
            nn.ReLU(True),

            nn.ConvTranspose2d(672, 336, 5, stride=2, padding=0),
            nn.BatchNorm2d(336),
            nn.ReLU(True),

            nn.ConvTranspose2d(336, 84, 5, stride=2, padding=0),
            nn.BatchNorm2d(84),
            nn.ReLU(True),

            nn.ConvTranspose2d(84, 1, 4, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_largerKernel_lessChannel(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_largerKernel_lessChannel, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(2, 64, 5, stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 256, 5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 5, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, 4, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_lessLayer(nn.Module): # does not work well
    def __init__(self, drop_rate):
        super(rdcnn_2_lessLayer, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(4, 84, 5, stride=2,padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(84, 168, 5, stride=2, padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(168, 84, 3, stride=1, padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),

            nn.ConvTranspose2d(84, 42, 3, stride=2, padding=0),
            nn.BatchNorm2d(42),
            nn.ReLU(True),

            nn.ConvTranspose2d(42, 1, 4, stride=2,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_largerKernel_lessChannel_1(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_largerKernel_lessChannel_1, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(2, 84, 5, stride=2,padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(84, 168, 5, stride=2, padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(168, 336, 5, stride=2, padding=1),
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 5, stride=2, padding=0),
            nn.BatchNorm2d(672),
            nn.ReLU(True),

            nn.ConvTranspose2d(672, 336, 5, stride=2, padding=0),
            nn.BatchNorm2d(336),
            nn.ReLU(True),

            nn.ConvTranspose2d(336, 84, 5, stride=2, padding=0),
            nn.BatchNorm2d(84),
            nn.ReLU(True),

            nn.ConvTranspose2d(84, 1, 4, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_largerKernel_lessChannel_2(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_largerKernel_lessChannel_2, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(2, 128, 5, stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(128, 256, 5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(256, 512, 5, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, 5, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 1, 4, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_largerKernel_lessChannel_Amir(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_largerKernel_lessChannel_Amir, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(2, 64, 5, stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 256, 5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 5, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, 4, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class rdcnn_2_5Kernel_2Channel_binaryClassification(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_5Kernel_2Channel_binaryClassification, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(2, 84, 5, stride=2,padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(84, 168, 5, stride=2, padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(168, 336, 5, stride=2, padding=1),
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 5, stride=2, padding=0),
            nn.BatchNorm2d(672),
            nn.ReLU(True),

            nn.ConvTranspose2d(672, 336, 5, stride=2, padding=0),
            nn.BatchNorm2d(336),
            nn.ReLU(True),

            nn.ConvTranspose2d(336, 84, 5, stride=2, padding=0),
            nn.BatchNorm2d(84),
            nn.ReLU(True),

            nn.ConvTranspose2d(84, 1, 4, stride=1,padding=0),
            nn.Sigmoid(),

        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv2d_lstm(nn.Module):
    def __init__(self, drop_rate):
        super(conv2d_lstm, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(2, 84, 5, stride=2,padding=1),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate),

            nn.Conv2d(84, 168, 5, stride=2, padding=1),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(168, 336, 5, stride=2, padding=1),
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 5, stride=2, padding=0),
            nn.BatchNorm2d(672),
            nn.ReLU(True),

            nn.ConvTranspose2d(672, 336, 5, stride=2, padding=0),
            nn.BatchNorm2d(336),
            nn.ReLU(True),

            nn.ConvTranspose2d(336, 84, 5, stride=2, padding=0),
            nn.BatchNorm2d(84),
            nn.ReLU(True),

            nn.ConvTranspose2d(84, 1, 4, stride=1,padding=0),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x