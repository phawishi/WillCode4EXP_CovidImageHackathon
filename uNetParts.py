import torch
import torch.nn as nn
import torch.nn.functional as F

globalSymSize=3

'''
******************************************************************************************
                                        Important notice
These uNetImplement.py and uNetParts.py for UNET architecture are not entirely mine.
These are code modified from a generous repository milesial/Pytorch-UNet/ who show me how 
to implement UNet

visit the link below for more info
https://github.com/milesial/Pytorch-UNet/tree/master/unet

These code are intended for educational purpose. The modification is for the Hackathon and
for people who what to study on how my team modified UNet for classification.

For any people who learn from my code, please also give credit to the original repository
when the credit is due.
******************************************************************************************
'''


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=globalSymSize, padding=1),
            nn.BatchNorm2d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=globalSymSize, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class TypDown(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=globalSymSize, padding=1),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    # implement attention here

    def __init__(self, in_channels, out_channels, nClasses=3):
        super(OutConv, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels, nClasses, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.forward0 = nn.Linear(out_channels, 1)
        self.forward1 = nn.Linear(out_channels, 1)
        self.forward2 = nn.Linear(out_channels, 1)
        self.lastSoftmax = nn.Softmax(dim=1)
        self.dropout0=torch.nn.Dropout(p=0.25)
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.25)

    def forward(self, x):
        attentionBinaryMask=self.mask(x)
        '''
        print(self.conv(x).shape)
        print(attentionBinaryMask[:, 0, :, :].shape)
        print('cehck up')
        exit()
        '''
        convOut0 = self.conv(x) * attentionBinaryMask[:, 0:1, :, :]
        convOut0 = torch.logsumexp(convOut0,(2 , 3))
        #convOut0=self.dropout0(convOut0)
        Out0=self.forward0(convOut0)

        convOut1 = self.conv(x) * attentionBinaryMask[:, 1:2, :, :]
        convOut1 = torch.logsumexp(convOut1, (2, 3))
        #convOut1 = self.dropout1(convOut1)
        Out1 = self.forward1(convOut1)

        convOut2 = self.conv(x) * attentionBinaryMask[:, 2:3, :, :]
        convOut2 = torch.logsumexp(convOut2, (2, 3))
        #convOut2 = self.dropout2(convOut2)
        Out2 = self.forward2(convOut2)

        #return self.conv(x)
        return self.lastSoftmax(torch.cat((Out0, Out1, Out2), dim=1))

    def getAttentionMap(self, x):
        attentionBinaryMask=self.mask(x)

        return attentionBinaryMask