# -*- coding: utf-8 -*-

import argparse
import yaml
from importlib import import_module
import time
import os

import torch

from utils.setseed import set_seed
from utils.utils import yaml_logger, make_dir, best_logger, save_model, MetricLogger
from utils.train_method import train_one_epoch, evaluate

from main.transform import get_transform
from main.dataset import MaskDataset
from main.dataloader import getDataloader
from main.loss import create_criterion


def run(args, cfg, device):
    set_seed(cfg['seed'])

    # cfg saved 폴더에 저장
    cfg['saved_dir'] = make_dir(cfg['saved_dir'], cfg['exp_name'])
    yaml_logger(args, cfg)

    # Transform 불러오기
    train_transform, val_transform = get_transform()

    # # DataSet 설정
    train_dataset = MaskDataset(cfg['train_path'], transform=train_transform)
    valid_dataset = MaskDataset(cfg['valid_path'], transform=val_transform)

    # # DataLoader 설정
    train_loader = getDataloader(dataset=train_dataset, **cfg['train_dataloader']['params'])
    valid_loader = getDataloader(dataset=valid_dataset, **cfg['valid_dataloader']['params'])

    # Model 불러오기
    model_module = getattr(import_module("main.model"), cfg['model']['name'])
    model = model_module(num_classes = cfg['model']['class']).to(device)
    
    # Loss function 설정
    if cfg['criterion']['weight']:
        class_weights = torch.FloatTensor(cfg['criterion']['weight'])
        criterion = create_criterion(cfg['criterion']['name'], weight=class_weights)
    else:
        criterion = create_criterion(cfg['criterion']['name'])

    # Optimizer 설정
    opt_module = getattr(import_module("torch.optim"), cfg['optimizer']['name'])
    optimizer = opt_module(params = model.parameters(), **cfg['optimizer']['params'])

    # Scheduler 설정
    scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfg['scheduler']['name'])
    scheduler=scheduler_module(optimizer, **cfg['scheduler']['params'])

    # 학습 파라미터 설정
    N_EPOCHS = cfg['epochs']
    start_time = time.time()

    classes = cfg['model']['class']
    cutmix_prob = cfg['cutmix_prob']

    # 학습
    best_f1 = 0

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_one_epoch(model, criterion, optimizer, scheduler, train_loader, device, epoch, cutmix_prob)
        metrics = evaluate(model, criterion, valid_loader, device=device)
        
        best_logger(cfg['saved_dir'], epoch, N_EPOCHS, metrics)

        if best_f1 < metrics[2]:
            best_f1 = metrics[2]

            checkpoint = model.state_dict()
            save_model(checkpoint, os.path.join(cfg['saved_dir'], "best_model.pt"))
    
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config/sample.yaml', help='yaml file path')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Yaml 파일에서 config 가져오기
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(args, cfg, device)