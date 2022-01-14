import torch
import numpy as np

def rand_bbox(size): # size : [Batch_size, Channel, Width, Height]
    W = size[2] # 512 // 2 = 256
    H = size[3] # 384 // 2 = 192

    # 패치의 중앙 좌표 값 cx, cy
    cx = W // 2 # 위아래 범위의 중앙
    cy = H // 2 # 좌우 범위의 중앙

    # 패치 모서리 좌표 값 
    bbx1 = 0 # 아래 = 0
    bby1 = np.clip(0, 0, H) # 왼쪽 = 0
    bbx2 = W # 위 = W
    bby2 = np.clip(cy, 0, H) # 오른쪽 = 좌우 범위의 중앙
    
    return bbx1, bby1, bbx2, bby2

def cutmix(model, criterion, images, labels, device):
    rand_index = torch.randperm(images.size()[0]).to(device)
    target_a = labels # 원본 이미지 label
    target_b = labels[rand_index] # 패치 이미지 label       
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size())
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    outs = model(images)
    loss = criterion(outs, target_a) * 0.5 + criterion(outs, target_b) * 0.5
    
    return outs, loss