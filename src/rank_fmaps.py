import argparse


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
parser.add_argument('--gan_name', type=str,  default='progan_val')
parser.add_argument('--have_classes', type=bool,  default=True)

# Images per class for identifying channels
parser.add_argument('--num_real', type=int,  default=5)
parser.add_argument('--num_fake', type=int,  default=5)

# Saving
parser.add_argument('--save_pt_files', type=bool,  default=True)

def main():
    args = parser.parse_args()

    if args.arch == 'resnet50':
        from fmap_ranking.rank_fmaps_ud_r50 import rank_fmaps

        rank_fmaps(args.dataset_dir, args.gan_name, 
        args.blur_jpg, args.have_classes, 
        args.bsize,
        args.num_real, args.num_fake,
        args.save_pt_files,
        normalize_only_using_positive = False,
        topk=None)

    elif args.arch == 'efb0':
        from fmap_ranking.rank_fmaps_ud_efb0 import rank_fmaps

        rank_fmaps(args.dataset_dir, args.gan_name, 
        args.blur_jpg, args.have_classes, 
        args.bsize,
        args.num_real, args.num_fake,
        args.save_pt_files,
        normalize_only_using_positive = False,
        topk=None)




if __name__=='__main__':
    main()