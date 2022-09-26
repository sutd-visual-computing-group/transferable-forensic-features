import argparse, os
import imp
import math
from utils.general import get_all_channels
from activation_histograms.get_activation_values import get_activation_values

import torch
from torchvision import models, transforms

from scipy import stats
# from scipy imistats import ktest

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Rank feature maps of universal detectors...')

# architecture
parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'efb0'])

# Data augmentation of model
parser.add_argument('--blur_jpg', type=str, required=True)

# dataset
parser.add_argument('--gan_name', type=str, nargs='+', default='progan_val')

# topk
parser.add_argument('--topk', type=int,  default=114)


def main():
    args = parser.parse_args()

    if(type(args.gan_name)==str):
        args.gan_name = [args.gan_name,]
    
    for gan_name in args.gan_name:
        csv_path = 'output/median_test/{}_{}/{}.csv'.format(args.arch, args.blur_jpg, gan_name)
        df = pd.read_csv(csv_path)

        # Only some channels
        df = df.head(args.topk)

        total_top_channels = df.shape[0]
        
        df = df [ df['p_value'] <= 0.05]
        print(df)
        total_color_channels = df.shape[0]

        print(total_top_channels, total_color_channels)
        print("color-conditional channel %", gan_name, (total_color_channels / total_top_channels)*100.0)





if __name__=='__main__':
    main()