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
from sklearn.metrics import f1_score
import os
import natsort
from PIL import Image

# for reading the testing image folder
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

img_folder_path_Train = './competeTrain/'
img_folder_path_Validate = './competeValidate/'
img_folder_path_Test = './competeTest/'
modelSaveFolder='./competeUnet_FullAttention_Model/'
totalEpoch=1 # number of consecutive epoch run
batch_size = 10
loadround=199#150#63 # number of epoch round to be skipped because it is already have trained model
validRun=True#True # True: no training only validation run; False: do training and validation

if(not os.path.exists(modelSaveFolder)):
    os.mkdir(modelSaveFolder)
size=(256,256)

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

train_dataset = tv.datasets.ImageFolder(root=img_folder_path_Train,transform=trsfm_Train)
#train_dataset = tv.datasets.ImageFolder(root=img_folder_path_Train,transform=trsfm_ValidateTest)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True);
validate_dataset = tv.datasets.ImageFolder(root=img_folder_path_Validate,transform=trsfm_ValidateTest)
validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size,shuffle=False);

my_dataset = CustomDataSet(img_folder_path_Test, transform=trsfm_ValidateTest)
test_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# use GPU, comment 'cuda' and uncomment 'cpu' for cpu version
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
    cumLoss=0
    sumCorrectClassify=0
    allClassified=0
    currentTime=time.time()

    # training run
    model.train()
    for idx, img in enumerate(train_loader):
        if(validRun):
            break
        inData=img[0]
        inLabel=img[1]

        dataTensor = inData.reshape(-1, 3, size[0], size[1]).float().to(device)
        inLabel = inLabel.type(torch.LongTensor)
        labelTensor = inLabel.reshape(-1).long().to(device)

        # Forward pass
        outTensor = model(dataTensor)
        loss = criterion(outTensor, labelTensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumLoss+=loss.data.cpu().numpy()

        predResult=outTensor.data.cpu().numpy()
        gtLabel=labelTensor.data.cpu().numpy()
        predResult=np.argmax(predResult,axis=1)
        sumCorrectClassify+=np.sum(predResult==gtLabel)
        allClassified+=len(gtLabel.tolist())


        #print('\t loss so far: {}'.format(cumLoss))
    if (not validRun):
        trainAcc = sumCorrectClassify / allClassified
        trainCumLoss = cumLoss
        # print('finish round', epoch)
        print('\ttotal train loss', trainCumLoss)
        print('\ttrainAcc', trainAcc)
        print('elapseTime', time.time() - currentTime)

        writetext = 'epoch{}, trainCumLoss: {} trainAcc: {}\n'.format(loadround+epoch,trainCumLoss,trainAcc)
        trainResultsFile = open("trainResults_UnetFullAttention.txt", "a")
        trainResultsFile.write(writetext)
        trainResultsFile.close()

        path = os.path.join(modelSaveFolder, 'modelround{}'.format(loadround+epoch+1))
        torch.save(model.state_dict(), path)
    else:
        path = os.path.join(modelSaveFolder, 'modelround{}'.format(loadround+epoch+1))
        netArch.load_state_dict(torch.load(path))


    cumLoss = 0
    sumCorrectClassify = 0
    allClassified = 0
    currentTime = time.time()
    # validate run
    model.eval()
    for idx, img in enumerate(validate_loader):
        inData = img[0]
        inLabel = img[1]

        dataTensor = inData.reshape(-1, 3, size[0], size[1]).float().to(device)
        inLabel = inLabel.type(torch.LongTensor)
        labelTensor = inLabel.reshape(-1).long().to(device)

        # Forward pass
        outTensor = model(dataTensor)
        loss = criterion(outTensor, labelTensor)

        cumLoss += loss.data.cpu().numpy()
        predResult = outTensor.data.cpu().numpy()
        gtLabel = labelTensor.data.cpu().numpy()
        predResult = np.argmax(predResult,axis=1)
        sumCorrectClassify += f1_score(gtLabel, predResult, average='weighted')#np.sum(predResult == gtLabel)
        allClassified += len(gtLabel.tolist())

        # print('\t loss so far: {}'.format(cumLoss))

    print('\ttotal valid loss', cumLoss)
    validAcc = sumCorrectClassify
    validCumLoss=cumLoss
    print('\tvalidAcc', validAcc)
    print('elapseTime', time.time() - currentTime)
    print('finish round', epoch)

    writetext = 'epoch{}, validCumLoss: {} validAcc: {}\n'.format(loadround+epoch, validCumLoss, validAcc)
    validResultsFile = open("validResults_UnetFullAttention.txt", "a")
    validResultsFile.write(writetext)
    validResultsFile.close()

    collectTestResult=[]
    collectDataName=[]
    for idx, img in enumerate(test_loader):
        inData = img[0]
        dataNameList = img[1]
        #dataName= img[1].split(',')
        #dataName=dataName[0]
        for i in range(len(img[1])):
            dataName = dataNameList[i]
            dataName = dataName.split('.')
            dataName = dataName[0]
            collectDataName.append(dataName)

        #inLabel = img[1]

        dataTensor = inData.reshape(-1, 3, size[0], size[1]).float().to(device)
        #inLabel = inLabel.type(torch.LongTensor)
        #labelTensor = inLabel.reshape(-1).long().to(device)

        # Forward pass
        outTensor = model(dataTensor)

        predResult = outTensor.data.cpu().numpy()
        predResult = np.argmax(predResult,axis=1)
        predResult = predResult.tolist()
        collectTestResult.extend(predResult)

        # print('\t loss so far: {}'.format(cumLoss))


    print(collectTestResult)

    # prepare submission file
    df = pd.DataFrame(data={"Id": collectDataName, "Predicted": collectTestResult})
    df.to_csv("./ForthTrialSub.csv", sep=',', index=False)
    print('######################################')


