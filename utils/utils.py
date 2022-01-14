from collections import defaultdict
import os
import shutil
import torch

# 디렉토리 생성
def make_dir(saved_dir, saved_name):
    path = os.path.join(saved_dir, saved_name)
    os.makedirs(path, exist_ok=True)

    return path

# yaml 파일 saved 폴더에 저장
def yaml_logger(args, cfg):
    file_name = f"{cfg['exp_name']}.yaml"
    shutil.copyfile(args.config, os.path.join(cfg['saved_dir'], file_name))

def best_logger(saved_dir, epoch, num_epochs, metrics):
    loss, accuracy, f1 = metrics
    with open(os.path.join(saved_dir, 'best_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"Epoch [{epoch}/{num_epochs}], Loss: {loss}, Accuracy: {accuracy}, F1 Score: {f1}\n")

def save_model(*args, **kwargs):
        torch.save(*args, **kwargs)

class MetricLogger:
    def __init__(self, num_iteration, header = ""):
        self.num_iteration = num_iteration
        self.metric  = defaultdict()
        self.header = header

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metric[key] = value

    def log(self, iteration):
        log_message = f"{self.header} [{iteration}/{self.num_iteration}]"
        for key, value in self.metric.items():
            log_message += f" {key} : {value:.3f}"

        print(log_message)