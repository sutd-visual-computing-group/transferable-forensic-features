import argparse, os
from turtle import position

from click import style
from sensitivity_assessment.color import all_metrics_sensitivity_analysis
import math

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

parser = argparse.ArgumentParser(description='Rank feature maps of universal detectors...')

# architecture
parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'efb0'])

# Data augmentation of model
#parser.add_argument('--blur_jpg', type=float, required=True, choices=[0.1, 0.5])

parser.add_argument('--blur_jpg', type=str, required=True)

# other metrics
parser.add_argument('--bsize', type=int,  default=16)

# dataset
parser.add_argument('--dataset_dir', type=str,  default='/mnt/data/v2.0_CNN_synth_testset/')
parser.add_argument('--gan_name', type=str, nargs='+', default='progan_val')
parser.add_argument('--have_classes', type=int,  default=1)

# Images per class for identifying channels
parser.add_argument('--num_instances', type=int,  default=5)



def main():
    args = parser.parse_args()

    if(type(args.gan_name)==str):
        args.gan_name = [args.gan_name,]
    
    args.have_classes = bool(args.have_classes)

    print(args.gan_name, args.have_classes)

    for gan_name in args.gan_name:
        (y_pred, y_true), (y_pred_grayscale, y_true_grayscale) = all_metrics_sensitivity_analysis(args.arch, 
        args.dataset_dir, gan_name, args.blur_jpg, 
        args.have_classes, args.bsize, num_instances=None)

        save_loc = os.path.join('output', 'whisker_plots', args.arch, str(args.blur_jpg))
        os.makedirs(save_loc, exist_ok=True)

        # Plot and save
        plot_box_whisker_plots(y_pred, y_true, y_pred_grayscale, y_true_grayscale, save_loc, gan_name)



def plot_box_whisker_plots(y_pred, y_true, y_pred_grayscale, y_true_grayscale, save_loc, gan_name):
    
    # Get GAN values
    y_pred_gan = y_pred[ y_true == 1 ]
    y_pred_gan_grayscale = y_pred_grayscale[ y_true_grayscale == 1 ]

    # sanity check
    ind1 = np.argwhere(y_true == 1 ).flatten()
    ind2 = np.argwhere(y_true_grayscale == 1 ).flatten()
    assert set(list(ind1)) == set(list(ind2))

    # # Focus only on samples with >= 20% prob
    # y_pred_gan = y_pred_gan[ y_pred_gan_grayscale >= 0.20 ]
    # y_pred_gan_grayscale = y_pred_gan_grayscale[ y_pred_gan_grayscale >= 0.20 ]
    # print(y_pred_gan.shape, y_pred_gan_grayscale.shape)

    # Focus only on samples with >= 20% prob
    y_pred_gan_grayscale = y_pred_gan_grayscale[ y_pred_gan >= 0.20 ]
    y_pred_gan = y_pred_gan[ y_pred_gan >= 0.20 ]
    print(y_pred_gan.shape, y_pred_gan_grayscale.shape)

    # Look at percentage
    y_pred_gan *= 100.0
    y_pred_gan_grayscale *= 100.0

    # Create plot
    plt.rcParams['axes.xmargin'] = 0
    plt.rc('font', weight='bold')
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)

    # Create data for boxplot
    data = [y_pred_gan, y_pred_gan_grayscale]
    m1, m2 = np.median(y_pred_gan), np.median(y_pred_gan_grayscale)
    #print(y_pred_gan.mean(), y_pred_gan_grayscale.mean(), (m1-m2)/2)
    
    # Define locations for annotations
    pos = [0.25, 0.40]
    y_loc = (m1-m2)/2 + m2 + 10
    y_loc = 50
    x_loc = (pos[0] + pos[1])/2 - 0.01
    #print( np.median(y_pred_gan), np.median(y_pred_gan_grayscale) )

    # Create boxplot
    bplot = ax.boxplot(data, showfliers=True, positions=pos, widths=[0.1, 0.1], labels=['Baseline', 'Grayscale'],
                manage_ticks=False, patch_artist=True, medianprops=dict(color="r", linewidth=5))
    #ax.axhline(m1, linestyle='--')
    #ax.axhline(m2, linestyle='--')
    
    # Annotate the drop in median prob
    text_for_drop = "{:.1f}%\ndrop".format((m1-m2))
    ax.annotate( '', xy=(x_loc, m2 ), xytext=(x_loc, m1 ) , horizontalalignment="center",
        arrowprops=dict(arrowstyle='<->',lw=5,  shrinkA = 0, shrinkB = 0, color='r') )
    ax.annotate( text_for_drop, xy=(x_loc, m2 ), xytext=(x_loc+0.05, y_loc) , horizontalalignment="center",
                color='r', weight='bold', fontsize=50)

    # Now make it aesthetic
    ax.set_xlim(0.19, 0.46)
    ax.set_xticks(pos)
    ax.set_xticklabels(['Baseline', 'Grayscale'], fontsize=50, weight='bold')
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(visible=True)
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel("Probability (%)", fontsize=50, weight='bold')
    plt.tight_layout()

    # Add colors
    colors = ['#83c5be', '#b7b7a4',  ]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # Save
    plt.savefig('{}/{}.pdf'.format(save_loc, gan_name), format='pdf', dpi=1200)
    #plt.show()



if __name__=='__main__':
    main()