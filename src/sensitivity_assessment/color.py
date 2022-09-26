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


def all_metrics_sensitivity_analysis(arch, parent_dir, gan_name, aug, have_classes, bsize, num_instances=None):
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

    transform_grayscale = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    # Obtain D
    if have_classes:
        dl = get_classwise_dataloader(root_dir, have_classes, num_instances , transform, bsize, onlyreal=False, onlyfake=False)
        dl_grayscale = get_classwise_dataloader(root_dir, have_classes, num_instances , transform_grayscale, bsize, onlyreal=False, onlyfake=False)

    else:
        dl = get_dataloader(root_dir, have_classes, num_instances , transform, bsize, onlyreal=False, onlyfake=False)
        dl_grayscale = get_dataloader(root_dir, have_classes, num_instances , transform_grayscale, bsize, onlyreal=False, onlyfake=False)
        dl = [dl]
        dl_grayscale = [ dl_grayscale ]

    df_ap = pd.DataFrame()
    df_uncalibrated_acc = pd.DataFrame()
    df_calibrated_acc = pd.DataFrame()

    # Calibrate GAN
    y_pred, y_true = get_probs(model, dl, device)
    original_calibrated_threshold = get_calibrated_thres(y_true, y_pred)

    # Original : Now get 2 sets of results for each setup (independent of topk)
    original_results, (y_pred, y_true) = get_ap_and_acc(model, device, dl, threshold=0.5)
    original_results_calibrated, (_, _) = get_ap_and_acc_with_new_threshold(y_pred=y_pred, y_true=y_true, threshold=original_calibrated_threshold, prefix='Baseline')

    # Grayscale : Now get 2 sets of results for each setup (independent of topk)
    grayscale_results, (y_pred_grayscale, y_true_grayscale) = get_ap_and_acc(model, device, dl_grayscale, threshold=0.5)
    grayscale_results_calibrated, (_, _) = get_ap_and_acc_with_new_threshold(y_pred=y_pred_grayscale, y_true=y_true_grayscale, threshold=original_calibrated_threshold, prefix='Grayscale')

    record_ap = {'gan_name': "{}".format(gan_name), 
                        'original_ap': original_results[0],
                        'grayscale_ap': grayscale_results[0],

                        }

    record_uncalibrated_acc = {'gan_name': "{}".format(gan_name), 

                        'original_r_acc': original_results[1],
                        'original_f_acc': original_results[2],
                        'original_acc': original_results[3],
                        'original_y_real_mean': original_results[4],
                        'original_y_fake_mean': original_results[5],

                        'original_y_real_std': original_results[6],
                        'original_y_fake_std': original_results[7],


                        'grayscale_r_acc': grayscale_results[1],
                        'grayscale_f_acc': grayscale_results[2],
                        'grayscale_acc': grayscale_results[3],
                        'grayscale_y_real_mean': grayscale_results[4],
                        'grayscale_y_fake_mean': grayscale_results[5],

                        'grayscale_y_real_std': grayscale_results[6],
                        'grayscale_y_fake_std': grayscale_results[7],

                        }


    record_calibrated_acc = {'gan_name/topk': "{}".format(gan_name), 
                        'original_threshold_calibrated' : original_calibrated_threshold,

                        'original_r_acc_calibrated': original_results_calibrated[1],
                        'original_f_acc_calibrated': original_results_calibrated[2],
                        'original_acc_calibrated': original_results_calibrated[3],

                        'grayscale_r_acc_calibrated': grayscale_results_calibrated[1],
                        'grayscale_f_acc_calibrated': grayscale_results_calibrated[2],
                        'grayscale_acc_calibrated': grayscale_results_calibrated[3],
    
                        }


    # Append to csv
    df_ap = df_ap.append(record_ap, ignore_index=True, sort=False)[list(record_ap.keys())]
    df_uncalibrated_acc = df_uncalibrated_acc.append(record_uncalibrated_acc, ignore_index=True, sort=False)[list(record_uncalibrated_acc.keys())]
    df_calibrated_acc = df_calibrated_acc.append(record_calibrated_acc, ignore_index=True, sort=False)[list(record_calibrated_acc.keys())]
    
    df_ap[df_ap.select_dtypes(include=['number']).columns] *= 100.0
    df_uncalibrated_acc[df_uncalibrated_acc.select_dtypes(include=['number']).columns] *= 100.0
    df_calibrated_acc[df_calibrated_acc.select_dtypes(include=['number']).columns] *= 100.0

    output_dir = os.path.join('output', 'grayscale-transfer', arch, '{}_{}'.format(gan_name, aug) )
    os.makedirs(output_dir, exist_ok=True)
    df_ap.to_csv('{}/{}-AP-#{}-samples-no-crop.csv'.format(output_dir, gan_name, dl[0].dataset.__len__()), index=None)
    df_uncalibrated_acc.to_csv('{}/{}-UNCALIBRATED_ACC-#{}-samples-no-crop.csv'.format(output_dir, gan_name, dl[0].dataset.__len__()), index=None)
    df_calibrated_acc.to_csv('{}/{}-CALIBRATED_ACC-#{}-samples-no-crop.csv'.format(output_dir, gan_name, dl[0].dataset.__len__()), index=None)
    
    return (y_pred, y_true), (y_pred_grayscale, y_true_grayscale)

if __name__=='__main__':
    pass

