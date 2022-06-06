import torch 
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from skimage.morphology import disk
from skimage.filters import rank
import os 
import numpy as np
import cv2 
import matplotlib.pyplot as plt 

# syp_before = ["preIRF","preSRF","prePED","preHRF"]
# syp_after = ["IRF","SRF","PED","HRF"]

"""
customized pytorch dataset
Each element of OCTimageDataset is a dictionary:
{"image": np.array , "label": int (0 or 1 or depth)}

image size: (596,1264,3) 
"""
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img

class RightCrop(object):
    """Crop the left part of the image in a sample.
    """
    def __call__(self, image):
        image = image[:496, image.shape[1]-768+128:image.shape[1]-128, :]
        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self,image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
              

class Normalize(object):
    """normalize images to fit them into the resnet50. """
    def __call__(self,image):
        # image, label = sample['image'], sample['label']
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

        for i in range(3):
            image[:,:,i] = (image[:,:,i]-mean[i])/std[i]
        
        # return {'image': image,
        #         'label': label}
        return image 

class ExtractRPE(object):
    """extract the RPE layer"""
    def __call__(self,image):
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold = 200/255
        image[image>=threshold] = 1
        image[image<threshold] = 0

        image_RPE = []
        for col in image.T:
            try:
                last_white_pixel = np.where(col == 1)[0].max()
                col_adj = np.zeros(len(col))
                col_adj[last_white_pixel] = 1
            except: 
                col_adj = np.zeros(len(col))
            image_RPE.append(col_adj)
        image_RPE = np.array(image_RPE)
        image = np.float32(image_RPE.T)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

class Remove(object):
    """remove the choroid part of the image"""
    def __call__(self,image):
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # fig, ax = plt.subplots(2,2)
        # ax[0][0].imshow(image,cmap = "gray")

        # 二值化提取
        binary_image = image.copy()
        threshold = 190/255
        binary_image[binary_image>=threshold] = 1
        binary_image[binary_image<threshold] = 0

        # ax[0][1].imshow(binary_image,cmap = "gray")

        # extract the line 
        image_RPE = []
        for col in binary_image.T:
            try:
                last_white_pixel = np.where(col == 1)[0].max()
                col_adj = np.zeros(len(col))
                col_adj[last_white_pixel] = 1
            except: 
                col_adj = np.zeros(len(col))
            image_RPE.append(col_adj)
        image_RPE = np.array(image_RPE)

        # ax[1][0].imshow(image_RPE.T,cmap = "gray")

        ## fit the line 
        try : 
            x = np.where(image_RPE == 1)[0]
            y = np.where(image_RPE == 1)[1]
            coef = np.polyfit(x,y,2)
            y_fit = np.polyval(coef,range(0,image_RPE.shape[0]))
            
        except : 
           y_fit = 200 + np.zeros(image_RPE.shape[0])

        fitted_map = []
        for i in range(image_RPE.shape[0]):
            a = np.zeros(image_RPE.shape[0])
            white_index = round(y_fit[i])-3
            try:
                a[white_index] = 1
            except: 
                a[200] = 1
            fitted_map.append(a)
        fitted_map = np.array(fitted_map)

        # ax[1][0].imshow(fitted_map.T,cmap = "gray")

        fill_image = []
        for col in fitted_map:
            new_line = np.zeros(len(col))
            # print(col.max())
            # print(np.where(col == 1))
            white_index = np.where(col== 1)[0][0]
            new_line[0:white_index] = 1
            fill_image.append(new_line)

        fill_image = np.array(fill_image).T
        upper_side = image*fill_image
        image = np.float32(upper_side)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # ax[1][1].imshow(image,cmap = "gray")

        return image

class Filt(object):
    """do mean filter on the image."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        footprint = disk(15)
        bilateral_result = rank.mean_bilateral(image,s0=500, s1=500,selem=footprint)
        bilateral_result = bilateral_result * (1/bilateral_result.max())
        bilateral_result[bilateral_result>=0.5] = 1
        bilateral_result[bilateral_result<0.5] = 0
        if bilateral_result.shape[-1] != 3:
            bilateral_result = np.stack((bilateral_result,) * 3, axis = -1)
        
        return {'image': bilateral_result,
                'label': label}

        