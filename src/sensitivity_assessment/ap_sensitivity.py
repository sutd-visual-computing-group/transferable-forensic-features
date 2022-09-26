# Import generic libraries
import os, sys, math, gc
import PIL

# Import scientific computing libraries
import numpy as np

# Import torch and dependencies
import torch
from torchvision import models, transforms

# Import other libraries
import matplotlib.pyplot as plt

# Import utils
from utils.heatmap_helpers import *
from utils.dataset_helpers import *
from utils.general import *


import copy
from collections import OrderedDict
import pandas as pd
import random, json
from tqdm import tqdm
from utils.mask_fmaps import *



def ap_sensitivity_analysis(arch, parent_dir, gan_name, aug, have_classes, bsize, num_instances=None, topk_list=None):
    ## ------------Set Parameters--------
    # Define device and other parameters
    device = torch.device('cuda:0')

    # model and weights
    if arch == 'resnet50':
        weightfn = './weights/resnet50/blur_jpg_prob{}.pth'.format(aug)
        model = get_resnet50_universal_detector(weightfn).to(device)
    
    elif arch == 'efb0':
        weightfn = './weights/efb0/blur_jpg_prob{}.pth'.format(aug)
        model = get_efb0_universal_detector(weightfn).to(device)

    root_dir = os.path.join(parent_dir, gan_name)

    # Dataset (Use same transforms as Wang et. al
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Obtain D
    if have_classes:
        dl = get_classwise_dataloader(root_dir, have_classes, num_instances , transform, bsize, onlyreal=False, onlyfake=False)

    else:
        dl = get_dataloader(root_dir, have_classes, num_instances , transform, bsize, onlyreal=False, onlyfake=False)
        dl = [dl]

    df_ap = pd.DataFrame()

    # Original : Independent of topk
    original_results, (y_pred, y_true) = get_ap_and_acc(model, device, dl, threshold=0.5)

    # Collect topk filters
    for topk in topk_list:
        topk_channels, lowk_channels, all_channels = get_all_channels(
            fake_csv_path="fmap_relevances/{}/{}_{}/{}-fake.csv".format(arch, gan_name, aug, gan_name), 
            topk=topk)

        print("Topk set to ", topk)
        assert topk == len(topk_channels), "mismatch"

        # Create model replicas by masking topk filters
        model_topk_masked, _ = mask_target_channels(copy.deepcopy(model), topk_channels)
        models_randomk_masked = [ mask_random_channels(copy.deepcopy(model), topk, topk_channels, all_channels )[0] for i in range(5)]
        model_lowk_masked, _ = mask_target_channels(copy.deepcopy(model), lowk_channels)

        # Topk : Now get 2 sets of results for each setup
        topk_masked_results, (y_pred, y_true)  = get_ap_and_acc(model_topk_masked, device, dl, 0.5)

        # Randomk : Now get 2 sets of results for each setup
        randomk_masked_results, (y_pred, y_true) = get_ap_and_acc_random(models_randomk_masked, device, dl, 0.5)

        # Lowk : Now get 2 sets of results for each setup
        lowk_masked_results, (y_pred, y_true) = get_ap_and_acc(model_lowk_masked, device, dl, 0.5)

        record_ap = {'gan_name/topk': "{}/{}".format(gan_name, topk), 

                            'original_ap': original_results[0],
                            'topk_masked_ap': topk_masked_results[0],
                            'randomk_masked_ap': randomk_masked_results[0],
                            'lowk_masked_ap': lowk_masked_results[0],

                            }

        df_ap = df_ap.append(record_ap, ignore_index=True, sort=False)[list(record_ap.keys())]

    
    output_dir = os.path.join('output', 'k_vs_ap', arch, "{}_{}".format( gan_name, aug))
    os.makedirs(output_dir, exist_ok=True)
    df_ap.to_csv('{}/{}-AP-#{}-samples-crop224.csv'.format(output_dir, gan_name, dl[0].dataset.__len__()), index=None)
       
    
