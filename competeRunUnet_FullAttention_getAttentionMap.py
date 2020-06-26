# training
import torch
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from uNetParts import *
from uNetImplement import *
#from processImageFolder import *
import time
import numpy as np
import pandas as pd
import cv2
import os
import natsort
from PIL import Image

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        imSize=image.size
        if(self.transform is not None):
            tensor_image = self.transform(image)
        else:
            tensor_image = image

        #return (tensor_image,int(idx>330))
        return (tensor_image, self.total_imgs[idx])

### these folder organization contain images only without subfolder
img_folder_path_Train = './competeTrainAll/'
img_folder_path_Validate = './competeValidateAll/'
img_folder_path_Test = './competeTestAll/'

modelSaveFolder='./competeUnet_FullAttention_Model/'

maskSavePathTrain='./competeTrainAll_attentionMask/'
maskSavePathValidate='./competeValidateAll_attentionMask/'
maskSavePathTest='./competeTestAll_attentionMask/'
if(not os.path.exists(maskSavePathTrain)):
    os.mkdir(maskSavePathTrain)
if(not os.path.exists(maskSavePathValidate)):
    os.mkdir(maskSavePathValidate)
if(not os.path.exists(maskSavePathTest)):
    os.mkdir(maskSavePathTest)

totalEpoch=1
batch_size = 1
loadround=145
validRun=True

if(not os.path.exists(modelSaveFolder)):
    os.mkdir(modelSaveFolder)
size=(256,256)
'''
trsfm_Train = tv.transforms.Compose([
        tv.transforms.Resize(size),
        tv.transforms.ToTensor()])
'''
trsfm_Train = tv.transforms.Compose([
    tv.transforms.Resize(size),
    tv.transforms.RandomHorizontalFlip(p=0.5),
    tv.transforms.RandomCrop((size[0]-6,size[1]-6)),
    tv.transforms.RandomAffine((-5,5), translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(-3,3)),

    tv.transforms.Resize(size),
    tv.transforms.ToTensor()])

trsfm_ValidateTest = tv.transforms.Compose([
        tv.transforms.Resize(size),
        tv.transforms.ToTensor()])

my_dataset = CustomDataSet(img_folder_path_Train, transform=trsfm_ValidateTest)
train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
my_dataset = CustomDataSet(img_folder_path_Validate, transform=trsfm_ValidateTest)
validate_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
my_dataset = CustomDataSet(img_folder_path_Test, transform=trsfm_ValidateTest)
test_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# comment/uncomment to select version cpu/gpu
device = torch.device('cuda')
#device = torch.device('cpu')
learningRate=0.0001
outFeature=128
rawChannel=3
nClasses=3
netArch=UNet_Full(rawChannel, outFeature, n_classes=nClasses)
if(loadround>0):
    path=os.path.join(modelSaveFolder,'modelround{}'.format(loadround))
    netArch.load_state_dict(torch.load(path))
model = netArch.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, eps=0.1)

for epoch in range(totalEpoch):
    for idx, img in enumerate(train_loader):
        inData=img[0]
        imgName=img[1]

        dataTensor = inData.reshape(-1, 3, size[0], size[1]).float().to(device)
        # Forward pass
        #outWeight0 = model.getClassWeight(0)
        #outWeight1 = model.getClassWeight(1)
        #outWeight2 = model.getClassWeight(2)

        attentionMaps=model.getAttentionMap(dataTensor)
        for i in range(4):
            map=attentionMaps[i]

            map = np.squeeze(map)
            map = np.transpose(map, (1, 2, 0))
            resizedMap = cv2.resize(map, (size))
            map = np.transpose(resizedMap, (2, 0, 1))
            map = np.expand_dims(map, 0)

            attentionMaps[i]=map

        for j in range(4):
            if(j ==1):
                map=attentionMaps[j]
            else:
                map=np.maximum(map,attentionMaps[j])

        attentionSave=map
        savePath=os.path.join(maskSavePathTrain,imgName[0])
        np.save(savePath, attentionSave)

    for idx, img in enumerate(validate_loader):
        inData=img[0]
        imgName=img[1]

        dataTensor = inData.reshape(-1, 3, size[0], size[1]).float().to(device)
        # Forward pass
        #outWeight0 = model.getClassWeight(0)
        #outWeight1 = model.getClassWeight(1)
        #outWeight2 = model.getClassWeight(2)

        attentionMaps=model.getAttentionMap(dataTensor)
        for i in range(4):
            map=attentionMaps[i]

            map = np.squeeze(map)
            map = np.transpose(map, (1, 2, 0))
            resizedMap = cv2.resize(map, (size))
            map = np.transpose(resizedMap, (2, 0, 1))
            map = np.expand_dims(map, 0)

            attentionMaps[i]=map

        for j in range(4):
            if(j ==1):
                map=attentionMaps[j]
            else:
                map=np.maximum(map,attentionMaps[j])

        attentionSave=map
        savePath=os.path.join(maskSavePathValidate,imgName[0])
        np.save(savePath, attentionSave)

    for idx, img in enumerate(test_loader):
        inData=img[0]
        imgName=img[1]

        dataTensor = inData.reshape(-1, 3, size[0], size[1]).float().to(device)
        # Forward pass
        #outWeight0 = model.getClassWeight(0)
        #outWeight1 = model.getClassWeight(1)
        #outWeight2 = model.getClassWeight(2)

        attentionMaps=model.getAttentionMap(dataTensor)
        for i in range(4):
            map=attentionMaps[i]

            map = np.squeeze(map)
            map = np.transpose(map, (1, 2, 0))
            resizedMap = cv2.resize(map, (size))
            map = np.transpose(resizedMap, (2, 0, 1))
            map = np.expand_dims(map, 0)

            attentionMaps[i]=map

        for j in range(4):
            if(j ==1):
                map=attentionMaps[j]
            else:
                map=np.maximum(map,attentionMaps[j])

        attentionSave=map
        savePath=os.path.join(maskSavePathTest,imgName[0])
        np.save(savePath, attentionSave)






