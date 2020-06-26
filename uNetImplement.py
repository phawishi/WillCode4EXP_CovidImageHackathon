import torch.nn.functional as F

from uNetParts import *

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

class UNet(nn.Module):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, n_channels, outFeature, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.outFeatureNum=outFeature

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, outFeature, nClasses=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_Full(nn.Module):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, n_channels, outFeature, n_classes, bilinear=True):
        super(UNet_Full, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.outFeatureNum=outFeature

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out4 = OutConv(64, outFeature, nClasses=n_classes)
        self.out3 = OutConv(128 // factor, outFeature, nClasses=n_classes)
        self.out2 = OutConv(256 // factor, outFeature, nClasses=n_classes)
        self.out1 = OutConv(512 // factor, outFeature, nClasses=n_classes)
        self.class0Weighting = torch.nn.Parameter(torch.ones(4,1))
        self.class1Weighting = torch.nn.Parameter(torch.ones(4,1))
        self.class2Weighting = torch.nn.Parameter(torch.ones(4,1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        logit1 = self.out1(x)
        x = self.up2(x, x3)
        logit2 = self.out2(x)
        x = self.up3(x, x2)
        logit3 = self.out3(x)
        x = self.up4(x, x1)
        logit4 = self.out4(x)


        class0Logit = torch.matmul(torch.cat((logit1[:, 0:1], logit2[:, 0:1], logit3[:, 0:1], logit4[:, 0:1]), dim=1),torch.abs(self.class0Weighting))
        class1Logit = torch.matmul(torch.cat((logit1[:, 1:2], logit2[:, 1:2], logit3[:, 1:2], logit4[:, 1:2]), dim=1),torch.abs(self.class1Weighting))
        class2Logit = torch.matmul(torch.cat((logit1[:, 2:3], logit2[:, 2:3], logit3[:, 2:3], logit4[:, 2:3]), dim=1),torch.abs(self.class2Weighting))

        #class0Logit = torch.sum(torch.cat((logit1[:, 0:1], logit2[:, 0:1], logit3[:, 0:1], logit4[:, 0:1]), dim=1),1,keepdim=True)
        #class1Logit = torch.sum(torch.cat((logit1[:, 1:2], logit2[:, 1:2], logit3[:, 1:2], logit4[:, 1:2]), dim=1),1,keepdim=True)
        #class2Logit = torch.sum(torch.cat((logit1[:, 2:3], logit2[:, 2:3], logit3[:, 2:3], logit4[:, 2:3]), dim=1),1,keepdim=True)

        logits=torch.cat((class0Logit, class1Logit, class2Logit), dim=1)
        return logits

    def getClassWeight(self, classNum):
        outData=None
        if(classNum==0):
            outData=self.class0Weighting.data.cpu().numpy()
        elif(classNum==1):
            outData = self.class1Weighting.data.cpu().numpy()
        elif (classNum == 2):
            outData = self.class2Weighting.data.cpu().numpy()
        return outData
    def getAttentionMap(self, x):
        outData=[]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        atten1 = self.out1.getAttentionMap(x)
        x = self.up2(x, x3)
        atten2 = self.out2.getAttentionMap(x)
        x = self.up3(x, x2)
        atten3 = self.out3.getAttentionMap(x)
        x = self.up4(x, x1)
        atten4 = self.out4.getAttentionMap(x)

        outData.append(atten1[:, :, :, :].data.cpu().numpy())
        outData.append(atten2[:, :, :, :].data.cpu().numpy())
        outData.append(atten3[:, :, :, :].data.cpu().numpy())
        outData.append(atten4[:, :, :, :].data.cpu().numpy())

        return outData


class TypicalCNN(nn.Module):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TypicalCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TypDown(n_channels, 64)
        self.down1 = TypDown(64, 128)
        self.down2 = TypDown(128, 256)
        self.down3 = TypDown(256, 512)
        #factor = 2 if bilinear else 1
        #self.down4 = Down(512, 1024 // factor)
        self.down4 = TypDown(512, 1024)
        '''
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        '''
        self.lin1=nn.Linear(65536,1024)
        self.lin2=nn.Linear(1024,128)
        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()

        #self.outLin=nn.Linear(128,128)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #print(x5.shape)
        #exit()
        #x5Flat = x5.view(-1,65536)
        x5Flat = torch.flatten(x5, start_dim=1)
        hidden1=self.lin1(x5Flat)
        hidden1=self.lrelu1(hidden1)
        #hidden1.sum().backward()
        #hidden1.sum().backward()
        hidden2=self.lin2(hidden1)
        hidden2=self.lrelu2(hidden2)
        #embedding=hidden1


        #logits = self.outc(x)
        return hidden2

