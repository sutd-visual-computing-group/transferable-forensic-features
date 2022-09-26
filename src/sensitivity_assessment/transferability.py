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


from utils.heatmap_helpers import *


from utils.dataset_helpers import *
from utils.general import *
import copy
from collections import OrderedDict
import pandas as pd
import random, json
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from tqdm import tqdm
from utils.mask_fmaps import *

from sklearn.metrics import roc_curve  

from termcolor import colored


def all_metrics_sensitivity_analysis(arch, parent_dir, gan_name, aug, have_classes, bsize, num_instances=None, topk_list=None):
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
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Obtain dataloaders
    if have_classes:
        dl = get_classwise_dataloader(root_dir, have_classes, num_instances , transform, bsize, onlyreal=False, onlyfake=False)

    else:
        dl = get_dataloader(root_dir, have_classes, num_instances , transform, bsize, onlyreal=False, onlyfake=False)
        dl = [dl]

    df_ap = pd.DataFrame()
    df_uncalibrated_acc = pd.DataFrame()
    df_calibrated_acc = pd.DataFrame()

    # Calibrate threshold
    y_pred, y_true = get_probs(model, dl, device)

    if gan_name == 'progan':
        original_calibrated_threshold = 0.5
    else:
        original_calibrated_threshold = get_calibrated_thres(y_true, y_pred)

    # Original : Now get 2 sets of results for each setup (independent of topk)
    print(colored('\n> Sensitivity assessments using feature map dropout', 'cyan'))
    original_results, (y_pred, y_true) = get_ap_and_acc(model, device, dl, threshold=0.5)
    original_results_calibrated , (_, _) = get_ap_and_acc_with_new_threshold(y_pred=y_pred, y_true=y_true, threshold=original_calibrated_threshold, prefix='baseline')

    # Collect topk filters
    for top_index, topk in enumerate(topk_list):
        #print("Topk set to ", topk)
        topk_channels, lowk_channels, all_channels = get_all_channels(
            fake_csv_path="fmap_relevances/{}/progan_val_{}/progan_val-fake.csv".format(arch, aug), 
            topk=topk)

        # Create model replicas by masking topk filters
        model_topk_masked, _ = mask_target_channels(copy.deepcopy(model), topk_channels)
        models_randomk_masked = [ mask_random_channels(copy.deepcopy(model), topk, topk_channels, all_channels )[0] for i in range(5)]
        model_lowk_masked, _ = mask_target_channels(copy.deepcopy(model), lowk_channels)

        print(colored('---------------------', 'cyan'))
        #print('---------------------')
        # Topk : Now get 2 sets of results for each setup
        topk_masked_results, (y_pred, y_true)  = get_ap_and_acc(model_topk_masked, device, dl, 0.5)
        topk_masked_results_calibrated , (_, _) = get_ap_and_acc_with_new_threshold(y_pred=y_pred, y_true=y_true, threshold=original_calibrated_threshold, prefix='top-k')

        # Randomk : Now get 2 sets of results for each setup
        randomk_masked_results, (y_pred, y_true) = get_ap_and_acc_random(models_randomk_masked, device, dl, 0.5)
        randomk_masked_results_calibrated  , (_, _) = get_ap_and_acc_with_new_threshold(y_pred=y_pred, y_true=y_true, threshold=original_calibrated_threshold, prefix='random-k')

        # Lowk : Now get 2 sets of results for each setup
        lowk_masked_results, (y_pred, y_true) = get_ap_and_acc(model_lowk_masked, device, dl, 0.5)
        lowk_masked_results_calibrated , (_, _) = get_ap_and_acc_with_new_threshold(y_pred=y_pred, y_true=y_true, threshold=original_calibrated_threshold, prefix='low-k')

        record_ap = {'gan_name/topk': "{}/{}".format(gan_name, topk), 

                            'original_ap': original_results[0],
                            'topk_masked_ap': topk_masked_results[0],
                            'randomk_masked_ap': randomk_masked_results[0],
                            'lowk_masked_ap': lowk_masked_results[0],

                            }

        record_calibrated_acc = {'gan_name/topk': "{}/{}".format(gan_name, topk), 
                            'original_threshold_calibrated' : original_calibrated_threshold,

                            'original_r_acc_calibrated': original_results_calibrated[1],
                            'original_f_acc_calibrated': original_results_calibrated[2],
                            'original_acc_calibrated': original_results_calibrated[3],
        
                            'topk_masked_r_acc_calibrated': topk_masked_results_calibrated[1],
                            'topk_masked_f_acc_calibrated': topk_masked_results_calibrated[2],
                            'topk_masked_acc_calibrated': topk_masked_results_calibrated[3],


                            'randomk_masked_r_acc_calibrated': randomk_masked_results_calibrated[1],
                            'randomk_masked_f_acc_calibrated': randomk_masked_results_calibrated[2],
                            'randomk_masked_acc_calibrated': randomk_masked_results_calibrated[3],


                            'lowk_masked_r_acc_calibrated': lowk_masked_results_calibrated[1],
                            'lowk_masked_f_acc_calibrated': lowk_masked_results_calibrated[2],
                            'lowk_masked_acc_calibrated': lowk_masked_results_calibrated[3],

                            }

        # Append to csv
        df_ap = df_ap.append(record_ap, ignore_index=True, sort=False)[list(record_ap.keys())]
        df_calibrated_acc = df_calibrated_acc.append(record_calibrated_acc, ignore_index=True, sort=False)[list(record_calibrated_acc.keys())]

    df_ap[df_ap.select_dtypes(include=['number']).columns] *= 100.0
    df_calibrated_acc[df_calibrated_acc.select_dtypes(include=['number']).columns] *= 100.0

    output_dir = os.path.join('output', 'transfer', arch, '{}_{}'.format(gan_name, aug) )
    os.makedirs(output_dir, exist_ok=True)
    df_ap.to_csv('{}/{}-AP-#{}-samples-crop.csv'.format(output_dir, gan_name, dl[0].dataset.__len__()), index=None)
    df_calibrated_acc.to_csv('{}/{}-CALIBRATED_ACC-#{}-samples-no-crop.csv'.format(output_dir, gan_name, dl[0].dataset.__len__()), index=None)
    

if __name__=='__main__':
    pass

