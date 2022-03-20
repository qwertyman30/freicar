import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import ConfusionMatrix

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from torchvision import transforms
import numpy as np
from torch.optim import SGD, Adam, lr_scheduler

from model import fast_scnn_model
from dataset_helper import freicar_segreg_dataloader
from dataset_helper import color_coder
from tqdm import tqdm
import matplotlib.pyplot as plt


def visJetColorCoding(name, img):
    img = img.detach().cpu().squeeze().numpy()
    color_img = np.zeros(img.shape, dtype=img.dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    cv2.imshow(name, color_img)


def visImage3Chan(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)


parser = argparse.ArgumentParser(description='Segmentation and Regression Training')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, help="Start at epoch X")
parser.add_argument('--batch_size', default=8, type=int, help="Batch size for training")
parser.add_argument('--num_epochs', default=50, type=int, help="Number of epochs for training")
parser.add_argument('--eval_freq', default=10, type=int, help="Evaluation frequency")
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 1000)')
parser.add_argument('--gamma_reg', default=10., type=float, help='Weighting for regression loss')
parser.add_argument('--gamma_seg', default=4., type=float, help='Weighting for segmentation loss')
best_iou = 0
args = parser.parse_args()


def main(resume, batch_size, num_epochs, start_epoch, eval_freq, print_freq, gamma_reg, gamma_seg):
    global best_iou

    # Create Fast SCNN model...
    model = fast_scnn_model.Fast_SCNN(3, 4)
    model = model.cuda()

    optimizer = Adam(model.parameters(), 5e-3)
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    seg_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.L1Loss()

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            model.load_state_dict(torch.load(resume))

        else:
            print("=> no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0

    # Data loading code
    load_real_images = False
    train_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                                            split='training', load_real=load_real_images)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=False, drop_last=True)

    train_loader_eval = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                    pin_memory=False, drop_last=True)

    eval_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                                           split='validation', load_real=load_real_images)

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=1,
                                              pin_memory=False, drop_last=False)

    best_loss = np.float('inf')
    train_losses_seg, train_losses_reg, train_losses = [], [], []
    validation_losses_seg, validation_losses_reg, validation_losses = [], [], []
    train_mious, val_mious = [], []
    for epoch in tqdm(range(start_epoch, num_epochs)):
        # train for one epoch
        train_loss_seg, train_loss_reg, train_loss = train(train_loader, model,
                                                           optimizer, scheduler, epoch,
                                                           seg_criterion, reg_criterion,
                                                           gamma_seg=gamma_seg, gamma_reg=gamma_reg,
                                                           print_freq=print_freq)
        train_losses_seg.append(train_loss_seg)
        train_losses_reg.append(train_loss_reg)
        train_losses.append(train_loss)

        val_loss_seg, val_loss_reg, val_loss = validate(eval_loader, model,
                                                        seg_criterion, reg_criterion,
                                                        gamma_seg=gamma_seg, gamma_reg=gamma_reg)
        validation_losses_seg.append(val_loss_seg)
        validation_losses_reg.append(val_loss_reg)
        validation_losses.append(val_loss)

        if epoch % eval_freq == 0 or epoch == (num_epochs - 1):
            train_ious = eval(train_loader_eval, model)
            val_ious = eval(eval_loader, model)
            train_miou = sum(train_ious.values()) / len(train_ious)
            val_miou = sum(val_ious.values()) / len(val_ious)
            train_mious.append(train_miou)
            val_mious.append(val_miou)

        # remember best iou and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
        }, False)
        if val_loss <= best_loss:
            save_model(model, f"model_{epoch}.pth")
            best_loss = val_loss

    return train_losses_seg, train_losses_reg, train_losses, validation_losses_seg, \
           validation_losses_reg, validation_losses, train_mious, val_mious, model,


def train(train_loader, model, optimizer, scheduler, epoch, seg_criterion, reg_criterion, gamma_seg, gamma_reg,
          print_freq=1000):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    total_loss_seg, total_loss_reg, total_loss = 0, 0, 0
    for i, (sample) in enumerate(train_loader):
        data_time.update(time.time() - end)

        image = sample['rgb'].cuda().float()
        lane_reg = sample['reg'].cuda().float()
        seg_ids = sample['seg'].cuda()

        ######################################
        # TODO: Implement me! Train Loop
        ######################################
        out = model(image)
        seg = out[0]
        reg = out[1]

        seg_loss = seg_criterion(seg, seg_ids.squeeze().long())
        reg_loss = reg_criterion(reg, lane_reg)
        total_loss_seg += seg_loss
        total_loss_reg += reg_loss

        loss = gamma_seg * seg_loss + gamma_reg * reg_loss
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    scheduler.step(epoch)
    return total_loss_seg, total_loss_reg, total_loss


def validate(loader, model, seg_criterion, reg_criterion, gamma_seg, gamma_reg):
    model.eval()

    total_loss_seg, total_loss_reg, total_loss = 0, 0, 0
    with torch.no_grad():
        for sample in loader:
            image = sample['rgb'].cuda().float()
            lane_reg = sample['reg'].cuda().float()
            seg_ids = sample['seg'].cuda()

            out = model(image)
            seg = out[0]
            reg = out[1]

            seg_loss = seg_criterion(seg, seg_ids.squeeze(0).long())
            reg_loss = reg_criterion(reg, lane_reg)
            total_loss_seg += seg_loss
            total_loss_reg += reg_loss

            loss = gamma_seg * seg_loss + gamma_reg * reg_loss
            total_loss += loss

        return total_loss_seg, total_loss_reg, total_loss


def eval(data_loader, model):
    model.eval()
    color_conv = color_coder.ColorCoder()
    color_coding = color_conv.color_coding
    classes = ['background', 'road', 'junction', 'car']
    class_dict = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in classes}
    miou_dict = {k: {'i': 0, 'u': 0} for k in classes}

    with torch.no_grad():
        for i, (sample) in enumerate(data_loader):

            image = sample['rgb'].cuda().float()
            lane_reg = sample['reg'].cuda().float()
            seg_ids = sample['seg'].cuda()

            out = model(image)
            seg = out[0]
            reg = out[1].cpu().squeeze(0).squeeze(0).numpy()
            preds = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy()

            seg_ids = seg_ids.cpu().numpy().squeeze(0).squeeze(0)
            cm = ConfusionMatrix(preds, seg_ids)
            cm_mat = cm.construct()
            total_miou = cm.computeMiou()

            for i, j in enumerate(classes):
                tp = cm_mat[i][i]
                fp = np.sum([cm_mat[i][k] for k in np.delete(np.arange(len(classes)), i)])
                fn = np.sum([cm_mat[k][i] for k in np.delete(np.arange(len(classes)), i)])
                class_dict[j]['tp'] += tp
                class_dict[j]['fp'] += fp
                class_dict[j]['fn'] += fn

            for i in class_dict.keys():
                miou_dict[i] = class_dict[i]['tp'] / (class_dict[i]['tp'] + class_dict[i]['fp'] + class_dict[i]['fn'])

        print(class_dict)
        print(miou_dict)
        return miou_dict


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_model(model, name):
    print("Saving model")
    torch.save(model.state_dict(), os.path.join("./saved_models", name))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


num_epochs = args.num_epochs
resume = args.resume
batch_size = args.batch_size
eval_freq = args.eval_freq
print_freq = args.print_freq
gamma_reg = args.gamma_reg
gamma_seg = args.gamma_seg
start_epoch = args.start_epoch

train_losses_seg, train_losses_reg, train_losses, validation_losses_seg, validation_losses_reg, \
    validation_losses, train_mious, val_mious, model = main(num_epochs=num_epochs, resume=resume, batch_size=batch_size,
                                                            start_epoch=start_epoch, eval_freq=eval_freq,
                                                            print_freq=print_freq, gamma_reg=gamma_reg,
                                                            gamma_seg=gamma_seg)

plt.figure(figsize=(25, 10))
plt.plot(train_losses_seg, label='Train Seg')
plt.plot(validation_losses_seg, label='Validation Seg')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Segmentation Losses vs epochs")
plt.legend(loc='upper right')
plt.savefig("seg_loss.png")
plt.show()

plt.plot(train_losses_reg, label='Train Reg')
plt.plot(validation_losses_reg, label='Validation Reg')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Regression Losses vs epochs")
plt.legend(loc='upper right')
plt.savefig("reg_loss.png")
plt.show()

plt.plot(train_losses, label='Train')
plt.plot(validation_losses, label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss vs epochs")
plt.legend(loc='upper right')
plt.savefig("Total_loss.png")
plt.show()

plt.plot(train_mious, label='Train')
plt.plot(val_mious, label='Validation')
plt.xlabel("Epoch(x10)")
plt.ylabel("MeanIoU")
plt.title("Mean ious vs epochs")
plt.legend(loc='upper right')
plt.savefig("ious.png")
plt.show()
