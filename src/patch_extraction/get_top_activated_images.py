
# Import base libraries
import os, sys, math, gc
import PIL
import copy
import random, json
from collections import OrderedDict

# Import scientific computing libraries
import numpy as np

# Import torch and dependencies
import torch
from torchvision import models, transforms

# Import utils
from utils.heatmap_helpers import *
from utils.dataset_helpers import *
from utils.general import *

# Import other libraries
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
# from scipy imistats import ktest

import seaborn as sns


# Keep penultimate features as global varialble such that hook modifies these features
penultimate_fts = None


def get_penultimate_fts(self, input, output):
    global penultimate_fts
    penultimate_fts = output
    return None


def get_class_specific_penultimate_fts(model, dls, device):
    global penultimate_fts
    penultimate_fts = None
    assert(penultimate_fts == None)

    model.eval() # Set model to eval mode
    m = torch.nn.AdaptiveMaxPool2d((1, 1)) # Look at maximum activation features

    # Store fts in a global array
    all_features = None
    fnames = []

    with torch.no_grad():
        for dl in dls:
            for batch_idx, data in enumerate(dl):
                x = data['image'].to(device)
                y = data['label']
                fname = data['filename']
                output = model(x)
                assert torch.is_tensor(penultimate_fts)

                fnames.extend(fname)

                if all_features is None:
                    all_features = (m(penultimate_fts.data.clone().cpu()) ).numpy().squeeze()
                
                else:
                    features = (m(penultimate_fts.data.clone().cpu()) ).numpy().squeeze()
                    all_features = np.concatenate((all_features, features), axis=0)

    return all_features, fnames



def get_activation_rankings(feature_map_name, arch, parent_dir, gan_name, aug, have_classes, bsize, num_instances=None):
    
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

    # Define device and other parameters
    feature_map_idx  = int(feature_map_name.split('.#')[-1].split('(')[0])
    layertobeattached = feature_map_name.split("#")[0][:-1]
    #print(layertobeattached)

    # Attach hook to original model
    new_handles = []
    for ind,(name,module) in enumerate(model.named_modules()):
      if name ==layertobeattached:
        print('name: {}'.format(name) )
        h=module.register_forward_hook( get_penultimate_fts )
        new_handles.append(h)

    # Use original transform as Wang et. al
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    #for idx, clss in enumerate(clsses):
    root_dir = os.path.join(parent_dir, gan_name)

    if have_classes:
        dls = get_classwise_dataloader(root_dir, have_classes, max_num=num_instances, transform=transform, bsize=bsize, onlyreal=False, onlyfake=True)
    else:
        dl = get_dataloader(root_dir, have_classes, max_num=num_instances, transform=transform, bsize=bsize, onlyreal=False, onlyfake=True)
        dls = [dl]

    clss_specific_fts_fake, fnames  = get_class_specific_penultimate_fts(model, dls, device)
    clss_specific_fts_fake = clss_specific_fts_fake[:, feature_map_idx]
    #print(clss_specific_fts_fake, fnames)

    # Sort and get the top activated images
    max_act_vals = np.asarray(clss_specific_fts_fake)
    idx = (-max_act_vals.copy()).argsort()

    # Save df
    df = pd.DataFrame()
    df['name'] = [fnames[i] for i in idx]
    df['max_act'] = [ max_act_vals[i] for i in idx]

    # Create output directory and save
    output_dir = os.path.join( 'output', 'activation_rankings', arch, '{}_{}'.format(gan_name, aug) )
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv("{}/{}.csv".format(output_dir, feature_map_name), index=None)
        



# if __name__=='__main__':

#     # Read fmaps
#     df = pd.read_csv("fmaps.csv")
#     fmaps = list(df['fmap'])[76:]

#     for i in range(len(fmaps)):
#         feature_map_name  = fmaps[i][:-1]
#         print(feature_map_name)
#         get_activation_rankings(feature_map_name)