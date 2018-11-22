import numpy as np
import cv2 as cv
import pandas as pd
import torch
import torch.utils.data

class ImageSteeringDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, is_train):
        self.df = pd.read_csv(csv_file)
        self.labels = self.df['steering'].values
        self.is_train = is_train


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        img, corr = self._choose_img(idx)
        img, corr = self._random_flip(img, corr)
        img = self._crop(img)
        img = self._resize(img, 200, 66)
        img = self._bgr2yuv(img)
        img = self._hwc2chw(img)
        img = self._2float(img)
        img = self._normalize(img)
        img = torch.from_numpy(img)

        label = self.labels[idx] + corr
        label = self._2float(label)
        label = torch.Tensor([label])

        return img, label

        
    def _choose_img(self, idx):
        if self.is_train == True:
            n = np.random.choice(3)
            if n == 0:
                img = cv.imread('./data/' + self.df['center'][idx].strip())
                corr = 0
            elif n == 1:
                img = cv.imread('./data/' + self.df['left'][idx].strip())
                corr = 0.2
            else:
                img = cv.imread('./data/' + self.df['right'][idx].strip())
                corr = -0.2
        else:
            img = cv.imread('./data/' + self.df['center'][idx].strip())
            corr = 0

        return img, corr


    def _random_flip(self, img, corr):
        if np.random.rand() < 0.5 and self.is_train == True:
            img = cv.flip(img, 1)
            corr = -corr
        return img, corr

    def _crop(self, img):
        return img[50:-20, :, :]

    def _resize(self, img, width, height):
        return cv.resize(img, (width, height), cv.INTER_AREA)

    def _bgr2yuv(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2YUV)

    def _hwc2chw(self, img):
        return np.transpose(img, (2, 0, 1))

    def _2float(self, img):
        return img.astype(np.float32)

    def _normalize(self, img):
        return img / 255.0
