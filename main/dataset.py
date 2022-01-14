import os
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, img_dir, transform):
        """
        img_dir: 학습 이미지 폴더(images)의 root directory('/opt/ml/input/data/train/images')
        transforms안넣으면 totensor변환만 해서 내보냄
        """
        self.transform = transform
        self.img_dir = img_dir
        
        self.path = []
        self.label = []
        self.indexs = [] # 추가
        self.groups = [] # 추가
        
        self.setup()

    
    def setup(self):
        cnt = 0 # 추가
        
        folder_list = os.listdir(self.img_dir)
        
        for folder_name in folder_list:
            if not folder_name.startswith("."):
                image_list = os.listdir(os.path.join(self.img_dir, folder_name))
                
                for image_name in image_list:
                    if not image_name.startswith('.'): 
                        self.path.append(f"{folder_name}/{image_name}")
                        
                        id, gender, race, age = folder_name.split('_')
                        
                        gender = 0 if gender == 'male' else 1
                        
                        age = int(age)
                        age_range = 0 if age < 30 else 1 if age < 60 else 2
                        
                        if 'incorrect' in image_name:
                            mask = 1
                        elif 'mask' in image_name:
                            mask = 0
                        elif 'normal' in image_name:
                            mask = 2
                            
                        self.label.append(mask * 6 + gender * 3 + age_range)
                        self.indexs.append(cnt) # 추가
                        self.groups.append(id) # 추가
                        cnt += 1 # 추가
                        
    
    def __getitem__(self, index):
        y = self.label[index]
        img_path = self.path[index]
    
        img = Image.open(os.path.join(self.img_dir,img_path))
        X = self.transform(image=np.array(img))['image']

        return X, y
    
    
    def __len__(self):
        return len(self.path)
        

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_paths[index]))

        if self.transform:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.img_paths)
