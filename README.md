# WillCode4EXP_CovidImageHackathon

This repository contains codes, scripts, and data used for team WillCode4EXP submission to AIAT Hackathon competing on classification of X-ray for Covid/Normal/Pneumonia classes. The method can be roughly described as modifying UNET+Attention model to classify visualize decision made with x-ray region.

## Important notice
These code are intended for educational purpose. The modification is for the Hackathon and
for people who what to study on how my team modified UNet for classification.

uNetImplement.py and uNetParts.py for UNET architecture are not entirely mine.
These are code modified from a generous repository milesial/Pytorch-UNet/ who show me how 
to implement UNet

visit the link below for more info
https://github.com/milesial/Pytorch-UNet/tree/master/unet

For any people who learn from my code, please also give credit to the original repository
when the credit is due.

## Dependecies
The codes were developed using Pytorch with following packages

numpy scipy pandas scikit-learn pillow flask opencv matplotlib scikit-image torchvision natsort

For installation using anaconda
```bash
conda install numpy scipy
conda install pandas scikit-learn
conda install -c pillow flask
conda install -c conda-forge opencv matplotlib scikit-image
```
## Code organization
<strong>competeRunUnet_FullAttention.py</strong> is main script for training validating and testing run. There is option to set number of epoch, load existing model, generate results .csv file, and etc.

<strong>competeRunUnet_FullAttention_getAttentionMap.py</strong> is a side script to extract attention maps and save into numpy array (.npy) for later visualization using Opencv

<strong>uNetImplement.py</strong> and <strong>uNetParts.py</strong> are network architecture implementation files.

<strong>opencvProcess.py</strong> is script for visualization with attention map.

<strong>ForthTrialSub.csv</strong> is submission used in the competition.

<strong>compete__</strong> are folders contain data for training, validation, and testing. The data in these folder are organized with subfolder indicating class label.

<strong>compete__All</strong> are data folders without any class label. The unorganized folders are easier to load in some scripts.

All the data has been manually crop to focus more on chest area.

## Usage
Generating submission .csv results from test images (CompeteTest folder).
```python
python competeRunUnet_FullAttention.py
```
competeRunUnet_FullAttention.py is automatically set to validate run be default, which mean there will be no training during the run.
Many part of codes has option to run with GPU/CPU version of Pytorch. Feel free to select the modes as appropriated

## Contributing
This is my first repository that I able to publicly release. Feel free to email me for questions/comments/suggestions at pthammas@uark.edu or phawis.hi@gmail.com. Pull request also appreciated.
