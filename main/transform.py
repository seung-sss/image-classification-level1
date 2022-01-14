import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.get_rgb_mean_std import get_img_mean_std_result

def get_transform():
    mean, std = get_img_mean_std_result()

    train_transform = A.Compose([
        A.Resize(int(512/2), int(384/2), p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        ], p=1.0)
    
    valid_transform = A.Compose([
        A.Resize(int(512/2), int(384/2), p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        ], p=1.0)

    return train_transform, valid_transform

def get_test_transform():
    mean, std = get_img_mean_std_result()
    
    test_transform = A.Compose([
        A.Resize(int(512/2), int(384/2), p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        ], p=1.0)
    
    return test_transform