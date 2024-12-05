import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from einops import rearrange, repeat
import time
import torch.optim as optim
import glob
import imageio
import numpy as np
import torch
from math import pi
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms
import argparse
import csv
import os
import cv2
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import utils
import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class ArgumentParser:
#     def add_argument(self, str, type, default):
#         setattr(self, str[2:], default)

#     def parse_args(self):
#         return self

def str_rec (names, data, unit=None, sep=', ', presets='{}'):
    if unit is None:
        unit = [''] * len(names)
    data = [str(i)[:6] for i in data]
    out_str = "{}: {{}} {{{{}}}}" + sep
    out_str *= len(names)
    out_str = out_str.format(*names)
    out_str = out_str.format(*data)
    out_str = out_str.format(*unit)
    out_str = presets.format(out_str)
    return out_str

def cifar(batch_size=64, size=32, path_to_data='../cifar_data'):
    """MNIST dataloader with (3, 28, 28) images.
    Parameters
    ----------
    batch_size : int
    size : int
        Size (height and width) of each image. Default is 28 for no resizing.
    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR10(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.CIFAR10(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader



# Implementation : dataset, metric
def aix(run, label_idx, model='1020-D', batch_size=16, size=(224,224)):
    mean_, std_ = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # params of ImageNet(pretrained)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),  # cv2로 읽은 이미지를 PIL 이미지로 변환
        transforms.Resize(size),
        transforms.RandomRotation(degrees=15),  # rotation: best performance
        transforms.ToTensor(),  # [0, 255] 범위의 uint8을 [0.0, 1.0] 범위의 float32로 변환
        transforms.Normalize(mean_, std_)
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),  # cv2로 읽은 이미지를 PIL 이미지로 변환
        transforms.Resize(size),
        transforms.ToTensor(),  # [0, 255] 범위의 uint8을 [0.0, 1.0] 범위의 float32로 변환
        transforms.Normalize(mean_, std_)
    ])
    path_img_crop = '/workspace/data/eye_disease_new/image_new/'
    path_label = '/workspace/data/eye_disease_new/'
    
    label_pair = pd.read_excel(path_label +'100_CAS_Label.xlsx')
    label_single = pd.read_excel(path_label +'1020_CAS_Label.xlsx')

    col_final = ['path', 'Red_lid_MG', 'Red_conj_MG', 'Swl_crncl_MG', 'Swl_lid_MG', 'Swl_conj_MG', 'pnum']

    col_pair_left = []
    col_pair_right = []
    for col in label_pair.columns:
        if ('Left' in col) and ('MG' in col):
            col_pair_left.append(col)
        if ('Right' in col) and ('MG' in col):
            col_pair_right.append(col)
            
    col_single_left = []
    col_single_right = []
    for col in label_single.columns:
        if ('left' in col) and ('MG' in col):
            col_single_left.append(col)
        if ('right' in col) and ('MG' in col):
            col_single_right.append(col)
    
    if label_idx in [1, 2, 4]:
        mode = 'eye'
    elif label_idx in [0, 3]:
        mode = 'eyelid'
    else:
        raise Exception
    y_dict_s = {} 
    for j in range(len(label_pair)):
        y_dict_s[int(label_pair.iloc[j]["p_num"])] = label_pair[col_pair_left].iloc[j, label_idx] or label_pair[col_pair_right].iloc[j, label_idx]
    name_split = 'split_pair_iter_win_' + str(label_idx) + '.pkl'
    with open(os.path.join('/workspace/code/CLUE_revise/split', name_split), 'rb') as f: 
        split_list_temp =  pickle.load(f)
    idx_train0, idx_val0 = train_test_split(
        split_list_temp[run][0], shuffle=True, stratify=[y_dict_s[p] for p in split_list_temp[run][0]], train_size=0.75)
    idx_train1, idx_val1 = train_test_split(
        split_list_temp[run][1], shuffle=True, stratify=[y_dict_s[p] for p in split_list_temp[run][1]], train_size=0.75)
    idx_train2, idx_val2 = train_test_split(
        split_list_temp[run][2], shuffle=True, stratify=[y_dict_s[p] for p in split_list_temp[run][2]], train_size=0.75)
    idx_train3, idx_val3 = train_test_split(
        split_list_temp[run][3], shuffle=True, stratify=[y_dict_s[p] for p in split_list_temp[run][3]], train_size=0.75)
    idx_train = np.hstack([idx_train0, idx_train1, idx_train2, idx_train3])
    idx_val = np.hstack([idx_val0, idx_val1, idx_val2, idx_val3])
    idx_test = split_list_temp[run][4]

    df_label_train, df_label_val, df_label_test = load_data(mode, idx_train, idx_val, idx_test, model = model)
    
    dataloader_train = torch.utils.data.DataLoader(
        Dataset_aix(
            data = df_label_train, transform = transform_train,  label_idx=label_idx), 
            batch_size = batch_size, shuffle = True, num_workers = 4)

    dataloader_val = torch.utils.data.DataLoader(
        Dataset_aix(
            data = df_label_val, transform = transform_test, label_idx=label_idx),
            batch_size = batch_size, shuffle = False, num_workers = 4)

    dataloader_test = torch.utils.data.DataLoader(
        Dataset_aix(
            data = df_label_test, transform = transform_test,  label_idx=label_idx),
            batch_size = batch_size, shuffle = False, num_workers = 4)

    return dataloader_train, dataloader_val, dataloader_test



def load_data(target, idx_train, idx_val, idx_test, model = '1020-D'):
    path_img_crop = '/workspace/data/eye_disease_new/image_new/'
    path_label = '/workspace/data/eye_disease_new/'
    label_pair = pd.read_excel(path_label +'100_CAS_Label.xlsx')
    label_single = pd.read_excel(path_label +'1020_CAS_Label.xlsx')
    col_final = ['path', 'Red_lid_MG', 'Red_conj_MG', 'Swl_crncl_MG', 'Swl_lid_MG', 'Swl_conj_MG', 'pnum']

    col_pair_left = []
    col_pair_right = []
    for col in label_pair.columns:
        if ('Left' in col) and ('MG' in col):
            col_pair_left.append(col)
        if ('Right' in col) and ('MG' in col):
            col_pair_right.append(col)
            
    col_single_left = []
    col_single_right = []
    for col in label_single.columns:
        if ('left' in col) and ('MG' in col):
            col_single_left.append(col)
        if ('right' in col) and ('MG' in col):
            col_single_right.append(col)
    

    if model == '1020-D':
        path_dslr_single = os.path.join(path_img_crop, 'DSLR', target)
        image_names = [name for name in os.listdir(path_dslr_single)
                                    if os.path.isfile(os.path.join(path_dslr_single, name))]
        image_names = sorted(image_names, key = lambda x: int(x.split(sep = '_')[0])) # sort
    elif model == '100-D':
        path_dslr_single = os.path.join(path_img_crop, 'DSLR_pair', target)
        image_names = [name for name in os.listdir(path_dslr_single)
                                   if (os.path.isfile(os.path.join(path_dslr_single, name)))]
        image_names = sorted(image_names, key = lambda x: int(x.split(sep = '_')[1])) # sort
    else: # Smartphone
        path_dslr_single = os.path.join(path_img_crop, 'Smartphone', target)
        image_names = [name for name in os.listdir(path_dslr_single)
                                   if (os.path.isfile(os.path.join(path_dslr_single, name)))] 
                                #    if (os.path.isfile(os.path.join(path_dslr_single, name))) and ('S21 Ultra' in name)] 
        image_names = sorted(image_names, key = lambda x: int(x.split(sep = '_')[1])) # sort
        image_names_df = pd.DataFrame([image_names], columns=[int(x.split(sep='_')[1]) for x in image_names])

    df_label_train = pd.DataFrame(columns = col_final)
    df_label_val = pd.DataFrame(columns = col_final)
    df_label_test = pd.DataFrame(columns = col_final)

    num_train = 0; num_val = 0; num_test = 0

    if model == '1020-D':
        for name in image_names:
            path = os.path.join(path_dslr_single, name)
            pnum = int(name.split(sep = '_')[0])
            if 'left' in name: # left eye
                temp = label_single[label_single['p_num'] == pnum][col_single_left]
            elif 'right' in name:
                temp = label_single[label_single['p_num'] == pnum][col_single_right]

            temp = temp.reset_index(drop = True)
            temp.columns = col_final[1:][:-1]
            temp['path'] = path
            temp['pnum'] = pnum
            temp = temp[col_final]

            if pnum in idx_train: # filtering by p_num
                df_label_train = pd.concat([df_label_train, temp], ignore_index = True)
                num_train += 1
            elif pnum in idx_val: 
                df_label_val = pd.concat([df_label_val, temp], ignore_index = True)
                num_val += 1                    
            elif pnum in idx_test: 
                df_label_test = pd.concat([df_label_test, temp], ignore_index = True)
                num_test += 1    
                    
    else: # Paired images: Smartphone or 100-D
        # idx_train 순서대로 데이터프레임 병합
        for name in image_names_df[idx_train].values[0]:
            path = os.path.join(path_dslr_single, name)
            pnum = int(name.split(sep='_')[1])
            if 'left' in name:  # left eye
                temp = label_pair[label_pair['p_num'] == pnum][col_pair_left]
            elif 'right' in name:
                temp = label_pair[label_pair['p_num'] == pnum][col_pair_right]
            temp.columns = col_final[1:][:-1]
            temp['path'] = path
            temp['pnum'] = pnum
            temp = temp[col_final]
            df_label_train = pd.concat([df_label_train, temp], ignore_index=True)
            num_train += 1

        for name in image_names_df.drop(idx_train,axis=1).values[0]:
            path = os.path.join(path_dslr_single, name)
            pnum = int(name.split(sep='_')[1])
            if 'left' in name:  # left eye
                temp = label_pair[label_pair['p_num'] == pnum][col_pair_left]
            elif 'right' in name:
                temp = label_pair[label_pair['p_num'] == pnum][col_pair_right]
            temp.columns = col_final[1:][:-1]
            temp['path'] = path
            temp['pnum'] = pnum
            temp = temp[col_final]
            if pnum in idx_val: 
                df_label_val = pd.concat([df_label_val, temp], ignore_index = True)
                num_val += 1                    
            elif pnum in idx_test: 
                df_label_test = pd.concat([df_label_test, temp], ignore_index = True)
                num_test += 1    

    return df_label_train, df_label_val, df_label_test

col_final = ['path', 'Red_lid_MG', 'Red_conj_MG', 'Swl_crncl_MG', 'Swl_lid_MG', 'Swl_conj_MG', 'pnum']


class Dataset_aix():
    def __init__(self, data, transform = None, label_idx=None):
        self.data = data
        self.list_path = self.data.iloc[:, 0].values
        self.list_label = self.data[col_final[1:-1]].values.astype(np.float32)
        self.transform = transform
        self.label_idx = label_idx
    
    def __len__(self):
        return len(self.list_path)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = cv2.cvtColor(cv2.imread(self.list_path[idx]),cv2.COLOR_BGR2RGB)
        if 'right' in self.list_path[idx]: # if right 
            image = cv2.flip(image, 1)
        if self.transform is not None:
            image = self.transform(image)

        target = torch.as_tensor(self.list_label[idx])
        target = target[self.label_idx].to(torch.int64)
        return image, target


def train(model, optimizer, trdat, tsdat, args):
    rec_names = ["iter", "loss", "f1", "nfe", "forwardnfe", "time/iter", "time"]
    rec_unit = ["","","","","","s","min"]
    itrcnt = 0
    loss_func = nn.CrossEntropyLoss()
    itr_arr = np.zeros(args.niters)
    loss_arr = np.zeros(args.niters)
    nfe_arr = np.zeros(args.niters)
    forward_nfe_arr = np.zeros(args.niters)
    time_arr = np.zeros(args.niters)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    outlist = []
    dataset_name = 'aix'
    os.makedirs(f'./results/{dataset_name}/tol/{args.model}/', exist_ok=True) 
    csvfile = open(f'./results/{dataset_name}/tol/{args.model}/{args.model}_{args.tol}_.csv', 'w')
    writer = csv.writer(csvfile)
    # training
    start_time = time.time()
    for epoch in range(1, args.niters+1):
        acc = 0
        dsize = 0
        iter_start_time = time.time()
        all_preds = []
        all_labels = []
        iter_start_time = time.time()
        for x, y in trdat:
            x = x.to(device=f'cuda:{args.gpu}')
            y = y.to(device=f'cuda:{args.gpu}')
            itrcnt += 1
            model[1].df.nfe = 0
            optimizer.zero_grad()
            # forward in time and solve ode
            pred_y = model(x)
            forward_nfe_arr[epoch - 1] += model[1].df.nfe
            # compute loss
            loss = loss_func(pred_y, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # make arrays
            itr_arr[epoch - 1] = epoch
            loss_arr[epoch - 1] += loss.detach()
            nfe_arr[epoch - 1] += model[1].df.nfe
            forward_nfe_arr[epoch - 1] += model[1].df.nfe
            # Collect predictions and labels
            pred_l = torch.argmax(pred_y, dim=1).cpu().numpy()
            all_preds.extend(pred_l)
            all_labels.extend(y.cpu().numpy())

        iter_end_time = time.time()
        time_arr[epoch - 1] = iter_end_time - iter_start_time
        loss_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        forward_nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt

        # Compute overall F1 score
        avg_f1 = f1_score(all_labels, all_preds, average="binary")  # For binary classification

        printouts = [epoch, loss_arr[epoch - 1], avg_f1, nfe_arr[epoch - 1], forward_nfe_arr[epoch - 1], time_arr[epoch - 1], (time.time() - start_time) / 60]
        print(str_rec(rec_names, printouts, rec_unit, presets="Train|| {}"))
        outlist.append(printouts)
        writer.writerow(printouts)
        
        # Test Loop
        if epoch % 2 == 0:
            model[1].df.nfe = 0
            test_start_time = time.time()
            all_preds = []
            all_labels = []
            loss = 0
            dsize = 0
            bcnt = 0
            for x, y in tsdat:
                # forward in time and solve ode
                dsize += y.shape[0]
                y = y.to(device=args.gpu)
                pred_y = model(x.to(device=args.gpu)).detach()
                pred_l = torch.argmax(pred_y, dim=1).cpu().numpy()
                all_preds.extend(pred_l)
                all_labels.extend(y.cpu().numpy())
                bcnt += 1
                # compute loss
                loss += loss_func(pred_y, y).detach() * y.shape[0]

            test_time = time.time() - test_start_time
            loss /= dsize
            # Compute overall F1 score for the test set
            avg_f1 = f1_score(all_labels, all_preds, average="binary")
            printouts = [epoch, loss.detach().cpu().numpy(), avg_f1, str(model[1].df.nfe / bcnt), None, test_time, (time.time() - start_time) / 60]
            print(str_rec(rec_names, printouts, presets="Test || {}"))

            outlist.append(printouts)
            writer.writerow(printouts)
    return outlist
