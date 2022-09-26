import argparse
from sensitivity_assessment.transferability import all_metrics_sensitivity_analysis
import math

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

# topk
parser.add_argument('--topk', type=int,  default=114)


def main():
    args = parser.parse_args()

    if(type(args.gan_name)==str):
        args.gan_name = [args.gan_name,]
    
    args.have_classes = bool(args.have_classes)

    print(args.gan_name, args.have_classes)

    for gan_name in args.gan_name:
        all_metrics_sensitivity_analysis(args.arch, 
        args.dataset_dir, gan_name, args.blur_jpg, 
        args.have_classes, args.bsize, num_instances=None, topk_list=[args.topk])


if __name__=='__main__':
    main()