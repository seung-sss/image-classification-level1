import os
import pandas as pd
import numpy as np
from PIL import Image


def get_img_mean_std(path):
    """
    RGB 평균 및 표준편차를 수집하는 함수입니다.
    Args:
        path: path_add_label.csv의 위치
    Returns:
        mean, std
    """
    img_info = dict(means=[], stds=[])
    path_label = pd.read_csv(path)
    for img_path in path_label['path']:
        img = np.array(Image.open(os.path.join('./train/train/images',img_path)))
        # (0,1,2)중 0번과 1번에 대해 평균
        # 512, 384 에 대한 평균이 3장 나옴
        img_info['means'].append(img.mean(axis=(0,1)))
        img_info['stds'].append(img.std(axis=(0,1)))
            
    return np.mean(img_info["means"], axis=0) / 255., np.mean(img_info["stds"], axis=0) / 255.

# 계산 시간 오래걸리므로 미리 저장해놓은 변수로 받기
def get_img_mean_std_result():
    mean = np.array([0.56019358, 0.52410121, 0.501457  ])
    std = np.array([0.23318603, 0.24300033, 0.24567522])
    return mean, std