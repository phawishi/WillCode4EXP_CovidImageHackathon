import torch
import numpy as np
import cv2
#import cv

# Load an color image in grayscale
selectImage=8
selectClassOverlay=0 # 0 covid 1 normal 2 pneumonia
imgNameList=['covid180.png','covid155.jpg','covid085.png','covid055.jpg','covid050.jpg','covid023.png','covid009.jpg','covid005.png','covid004.png','covid003.png']
#imgNameList=['pneumonia198.png','pneumonia170.png','pneumonia145.png','pneumonia070.png']
img = cv2.imread('./competeTrainAll/'+imgNameList[selectImage],cv2.IMREAD_COLOR)
mask=np.load('./competeTrainAll_attentionMask/'+imgNameList[selectImage]+'.npy')
mask=mask[0,selectClassOverlay,:,:];
#print(mask.shape)
#exit()
small = cv2.resize(img, (256,256))
#print(small.shape)
#print(small*mask)
#small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
#mask = (mask*255).astype(np.uint32)
im = np.array(mask * 255, dtype = np.uint8)
heatSmall=cv2.applyColorMap(im, cv2.COLORMAP_JET)
fin = cv2.addWeighted(heatSmall, 0.2, small, 0.8, 0)
cv2.imshow(imgNameList[selectImage], fin)

cv2.waitKey(0)

cv2.destroyAllWindows()