import argparse
import os
import math
from utils.general import get_all_channels
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Rank feature maps of universal detectors...')

# architecture
parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'efb0'])

# Data augmentation of model
# parser.add_argument('--blur_jpg', type=float, required=True, choices=[0.1, 0.5])
parser.add_argument('--blur_jpg', type=str, required=True)

# dataset
parser.add_argument('--gan_name', type=str, nargs='+', default='progan_val')

# Images per class for identifying channels
parser.add_argument('--num_instances', type=int,  default=5)




def create_collage(path_list, save_loc, gan, name):
    imgs = [ Image.open(i).resize((224, 224), Image.NEAREST) for i in path_list ][:5]
    #print(imgs)

    num_images = 5
    new_im = Image.new('RGB', (int(num_images*224), 224) )

    # Add border, convert back to pil
    

    index = 0
    for i in range(0, int(num_images*224), 224):
        patch_np = np.asarray(imgs[i//224])
        patch = cv2.copyMakeBorder(patch_np, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        patch = cv2.resize(patch, (224, 224))
        patch = Image.fromarray(patch)

        new_im.paste(patch, (i, 0))

    new_im.save('{}/{}.pdf'.format(save_loc, gan), quality=95, subsampling=0)



def main():
    args = parser.parse_args()

    if(type(args.gan_name)==str):
        args.gan_name = [args.gan_name,]
    
    #args.have_classes = bool(args.have_classes)
    print(args.gan_name)
    
    # Get feature map names
    topk_channels, lowk_channels, all_channels = get_all_channels(
            fake_csv_path="fmap_relevances/{}/progan_val_{}/progan_val-fake.csv".format(args.arch, args.blur_jpg), 
            topk=27)

    for gan_name in args.gan_name:
        for feature_map_name in topk_channels:
            # feature_map_idx  = int(feature_map_name.split('.#')[-1].split('(')[0])
            # layertobeattached = feature_map_name.split("#")[0][:-1]
            feature_map_name = feature_map_name.split('(')[0]
            
            patch_parent_path = os.path.join('output', 'patches', args.arch, gan_name, 'fake', feature_map_name, 'visualization')
            print(patch_parent_path)
            paths = [ os.path.join(patch_parent_path, i) for i in os.listdir(patch_parent_path) ]
            paths.sort()

            save_loc = os.path.join('output', 'collages', args.arch, str(args.blur_jpg), feature_map_name)
            os.makedirs(save_loc, exist_ok=True)
            create_collage(paths, save_loc, gan_name, feature_map_name)




if __name__=='__main__':
    main()