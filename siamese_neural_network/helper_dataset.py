# import required libraries
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.utils.data.sampler import SubsetRandomSampler 

import os
import numpy as np
import random
from PIL import Image


class OmniglotDataset(Dataset):
    
    def __init__(self, categories, root_dir, data_size, transform=None): 
        self.categories = categories
        self.root_dir = root_dir 
        self.transform = transform
        self.data_size = data_size 
  
    def __len__(self):
        return self.data_size 
  
    def __getitem__(self, idx):
        img1, img2, label = None, None, None
        label = np.array([label], dtype=np.float32) 
    
        if idx%2==0:
            category = random.choice(self.categories)
            character = random.choice(category[1])
            img_dir = self.root_dir + category[0] + "/" + character
            img1_name = random.choice(os.listdir(img_dir))
            img2_name = random.choice(os.listdir(img_dir))
            img1 = Image.open(img_dir + "/" + img1_name)
            img2 = Image.open(img_dir + "/" + img2_name)
            label = 1.0
    
        else:
            category1 = random.choice(self.categories)
            category2 = random.choice(self.categories)
            character1 = random.choice(category1[1])
            character2 = random.choice(category2[1])
            img_dir1 = self.root_dir + category1[0] + "/" + character1
            img_dir2 = self.root_dir + category2[0] + "/" + character2 
            img1_name = random.choice(os.listdir(img_dir1))
            img2_name = random.choice(os.listdir(img_dir2))
      
            while img1_name == img2_name:
              img2_name = random.choice(os.listdir(img_dir2))
              
            label = 0.0
            img1 = Image.open(img_dir1 + "/" + img1_name)
            img2 = Image.open(img_dir2 + "/" + img2_name)
    
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
  
        return img1, img2, label


class nWayOneShotValidSet(Dataset):
    def __init__(self, categories, root_dir, data_size, n_way, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.data_size = data_size 
        self.n_way = n_way 
        self.transform = transform 
    
    def __len__(self):
        return self.data_size 
  
    def __getitem__(self, idx):
        category = random.choice(self.categories)
        character = random.choice(category[1])
        img_dir = self.root_dir + category[0] + "/" + character
        img_name = random.choice(os.listdir(img_dir)) 
        main_img = Image.open(img_dir + "/" + img_name)
        if self.transform:
            main_img = self.transform(main_img)
    
        test_set = []
        label = np.random.randint(self.n_way)
        label = torch.from_numpy(np.array([label], dtype=int))
        for i in range(self.n_way):
            test_img_dir = img_dir 
            test_img_name = ""
            # find a random image from the from the same set as the main
            if i == label:
                test_img_name = random.choice(os.listdir(img_dir))
            else:
                test_category = random.choice(self.categories)
                test_character = random.choice(test_category[1])
                test_img_dir = self.root_dir + test_category[0] + "/" + test_character 
                while test_img_dir == img_dir:
                    test_img_dir = self.root_dir + test_category[0] + "/" + test_character 
                test_img_name = random.choice(os.listdir(test_img_dir))
                
            test_img = Image.open(test_img_dir + "/" + test_img_name)

            if self.transform:
                test_img = self.transform(test_img)
            test_set.append(test_img)

        return main_img, test_set, label 
