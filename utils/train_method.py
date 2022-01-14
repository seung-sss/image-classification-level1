import torch
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import f1_score

from main.cutmix import cutmix
from utils.utils import MetricLogger

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, cutmix_prob):
    num_iteration = len(data_loader)
    header = f"Epoch: [{epoch}]"
    metric_logger = MetricLogger(num_iteration, header)

    running_loss = 0.
    running_acc = 0.
    running_f1 = 0.
    n_iter = 0

    model.train()
    for i, (images, labels) in enumerate(tqdm(data_loader, leave=False)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # cutmix 적용
        if np.random.random() > cutmix_prob:
            output, loss = cutmix(model, criterion, images, labels, device)
        # cutmix 적용되지 않는 경우
        else:
            output = model(images)
            loss = criterion(output, labels)

        _, preds = torch.max(output, 1)

        loss.backward() 
        optimizer.step()

        running_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        n_iter += 1
        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)

        lr_scheduler.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_acc / len(data_loader.dataset)
    epoch_f1 = running_f1 / n_iter

    metric_logger.update(Loss=epoch_loss, F1=epoch_f1, Accuracy=epoch_acc)
    metric_logger.log(i)


def evaluate(model, criterion, data_loader, device):
    num_iteration = len(data_loader)
    header = "Test:"
    metric_logger = MetricLogger(num_iteration, header)

    running_loss = 0.
    running_acc = 0.
    running_f1 = 0.
    n_iter = 0

    model.eval()
    with torch.inference_mode():
        for i, (images, labels) in enumerate(tqdm(data_loader, leave=False)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)

            running_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            n_iter += 1
            running_loss += loss.item() * images.size(0)
            running_acc += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_acc / len(data_loader.dataset)
        epoch_f1 = running_f1 / n_iter

    metric_logger.update(Loss=epoch_loss, F1=epoch_f1, Accuracy=epoch_acc)
    metric_logger.log(i)

    return (epoch_loss, epoch_acc, epoch_f1)
