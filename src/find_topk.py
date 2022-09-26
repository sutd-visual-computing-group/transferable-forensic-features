import argparse
from sensitivity_assessment.ap_sensitivity import ap_sensitivity_analysis
import math

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
parser.add_argument('--num_instances', type=int,  default=5)

# topk
parser.add_argument('--topk', type=int,  default=5)


def main():
    args = parser.parse_args()
    topk_list = generate_bissect_candidates(args.topk, steps=1)
    print("Generated topk search list using bissected intervals => ", topk_list)

    topk_list = [args.topk]
    ap_sensitivity_analysis(args.arch, 
    args.dataset_dir, args.gan_name, args.blur_jpg, 
    args.have_classes, args.bsize, num_instances=args.num_instances, topk_list=topk_list)



def generate_bissect_candidates(topk, steps):
    list_of_vals = [0, topk]

    for i in range(steps):
        num_intervals = len(list_of_vals) - 1
        new_vals = [ int(math.ceil((list_of_vals[j] + list_of_vals[j+1])/2)) for j in range(num_intervals) ]
        list_of_vals.extend(new_vals)
        list_of_vals.sort()
        

    return list_of_vals


if __name__=='__main__':
    main()