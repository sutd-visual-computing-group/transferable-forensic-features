import argparse
import imp
import math
from utils.general import get_all_channels

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
parser.add_argument('--topk', type=int,  default=117)


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

    # No need to get for all channels, just get for color channels
    if args.arch == 'resnet50':

        from patch_extraction.extract_lrp_max_patches_using_filenames_resnet50 import get_high_activation_patches
        for gan_name in args.gan_name:
            for feature_map_name in topk_channels:
                get_high_activation_patches(feature_map_name, args.arch, gan_name, args.blur_jpg, args.bsize, num_instances=args.num_instances)

    elif args.arch == 'efb0':
        
        from patch_extraction.extract_lrp_max_patches_using_filenames_efb0 import get_high_activation_patches
        for gan_name in args.gan_name:
            for feature_map_name in topk_channels:
                get_high_activation_patches(feature_map_name, args.arch, gan_name, args.blur_jpg, args.bsize, num_instances=args.num_instances)


if __name__=='__main__':
    main()