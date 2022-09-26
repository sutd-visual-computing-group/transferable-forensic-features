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
# parser.add_argument('--blur_jpg', type=float, required=True, choices=[0.1, 0.5])
parser.add_argument('--blur_jpg', type=str, required=True)

# other metrics
parser.add_argument('--bsize', type=int,  default=16)

# dataset
parser.add_argument('--dataset_dir', type=str,  default='/mnt/data/v2.0_CNN_synth_testset/')
parser.add_argument('--gan_name', type=str, nargs='+', default='progan_val')
parser.add_argument('--have_classes', type=int,  default=1)

# Images per class for identifying channels
parser.add_argument('--num_instances', type=int,  default=5)

# topk
parser.add_argument('--topk', type=int,  default=114)


def plot_act_hist(feature_map_name, hist1, hist2, label1, label2, gan_name, save_dir):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['lines.linewidth'] = 10.0 # thicker lines
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)

    sns.kdeplot(data=hist1, ax=ax, label=label1)
    sns.kdeplot(data=hist2, ax=ax, label=label2)

    #plt.legend()
    # plt.xticks(linspace(0.0, 6.0, step=.5))
    # #plt.ylim(0, 1.0)
    # plt.title(feature_map_name, fontsize=60)
    # plt.tight_layout()

    #ax.legend(loc="best", bbox_to_anchor=(1.0, 0.9), prop={'size': 44})
    ax.set_title(feature_map_name.split('(')[0], fontsize=72, weight='bold')
    #ax.set_xticks(linspace(0.0, 6.0, step=2.0))
    ax.set_ylabel("density", fontsize=72, weight='bold')
    ax.xaxis.set_tick_params(labelsize=68)
    ax.yaxis.set_tick_params(labelsize=68)
    #ax.set_xlim(0, 5)
    x_ticks = ax.xaxis.get_major_ticks()
    x_ticks[0].label1.set_visible(False)

    ax.grid(True)

    ax.set_xlabel("max spatial activation",  fontsize=72, weight='bold' )
    fig.tight_layout()
    #plt.show()

    fig.savefig("{}/{}.pdf".format(save_dir, gan_name), format="pdf", dpi=1200, bbox_inches='tight')

    # Plot legend seperately
    figsize = (20, 0.1)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)

    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc="upper center", mode = "expand", 
                    ncol = 2, frameon=False, fontsize=50)

    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig('{}/legend.pdf'.format(save_dir), format='pdf', dpi=1200, bbox_inches='tight')
    #plt.show()
    plt.close()


def main():
    args = parser.parse_args()

    if(type(args.gan_name)==str):
        args.gan_name = [args.gan_name,]
    
    args.have_classes = bool(args.have_classes)

    print(args.gan_name, args.have_classes)

    # Get feature map names
    topk_channels, lowk_channels, all_channels = get_all_channels(
            fake_csv_path="fmap_relevances/{}/progan_val_{}/progan_val-fake.csv".format(args.arch, args.blur_jpg), 
            topk=args.topk)

    transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    transform_grayscale = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    print(topk_channels)
    
    for gan_name in args.gan_name:
         # Store median test results
        df_median_test = pd.DataFrame(columns=['feature_map_name', 'p_value'])

        for feature_map_name in topk_channels:
            df, act_fake = get_activation_values(feature_map_name, args.dataset_dir, args.arch, gan_name, args.have_classes, args.blur_jpg, args.bsize, 
            transform, num_instances=500)

            df, act_fake_grayscale = get_activation_values(feature_map_name, args.dataset_dir, args.arch, gan_name, args.have_classes, args.blur_jpg, args.bsize, 
            transform_grayscale, num_instances=500)

            # Perform median test, returned tuple in stat, p, median, contingency table
            median_test = stats.median_test(act_fake, act_fake_grayscale) [1]

            record = {'feature_map_name': feature_map_name , 'p_value': median_test}
            print(record)
            df_median_test = df_median_test.append(record, ignore_index=True)

            save_loc = os.path.join('output', 'activation_histograms', args.arch, str(args.blur_jpg), feature_map_name)
            os.makedirs(save_loc, exist_ok=True)

            plot_act_hist(feature_map_name, act_fake, act_fake_grayscale, "Baseline", "Grayscale", gan_name, save_dir=save_loc)


        output_dir= 'output/median_test/{}_{}'.format(args.arch, args.blur_jpg)
        os.makedirs(output_dir, exist_ok=True)
        df_median_test.to_csv("{}/{}.csv".format(output_dir, gan_name), index=False)

if __name__=='__main__':
    main()