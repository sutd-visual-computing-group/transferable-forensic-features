
# Import base libraries
import os, sys, math
import PIL

# Import scientific and plotting libraries
import numpy as np
import matplotlib.pyplot as plt

# Import torch libraries
import torch
from torchvision import models, transforms


def _get_all_paths(image_parent_dir, real_or_fake, have_classes, num_instances):
    """
    This is an internal function. Do not use it explicitly
    """
    all_img_paths = []
    
    if have_classes:
        classes = os.listdir(image_parent_dir)
        
        for c in classes:
            img_dir_location = os.path.join(image_parent_dir, c, real_or_fake)
            img_paths = [ os.path.join(img_dir_location, i) for i in os.listdir(img_dir_location) ]
            img_paths.sort()
            all_img_paths.extend(img_paths[:num_instances])

        #print(len(all_img_paths))
        #print(all_img_paths)
        return all_img_paths

    else:
        img_dir_location = os.path.join(image_parent_dir, real_or_fake)
        img_paths = [ os.path.join(img_dir_location, i) for i in os.listdir(img_dir_location) ]
        img_paths.sort()
        all_img_paths.extend(img_paths[:num_instances])

        #print(len(all_img_paths))
        #print(all_img_paths)
        return all_img_paths



class dataset_gan_vs_real(torch.utils.data.Dataset):
    """
    Class for binary classification for GAN and real images

    The labels used for training the original classifier are:
        0 : Real images
        1 : GAN/ fake images
    """
    def __init__(self, root_dir, have_classes, max_num, transform=None, onlyreal=False, onlyfake=False):
        self.classsubpaths =['0_real','1_fake']

        self.root_dir = root_dir
        self.transform = transform
        self.imgfilenames=[]
        self.labels =[]
        self.num_real_images = -1
        self.num_fake_images = -1

        real_img_paths = _get_all_paths(self.root_dir, self.classsubpaths[0], have_classes, max_num)
        fake_img_paths = _get_all_paths(self.root_dir, self.classsubpaths[1], have_classes, max_num)

        if onlyreal:
            self.imgfilenames.extend(real_img_paths)
            self.labels.extend([0]*len(real_img_paths))
            self.num_real_images = len(real_img_paths)

        elif onlyfake:
            #print(fake_img_paths)
            self.imgfilenames.extend(fake_img_paths)
            self.labels.extend([1]*len(fake_img_paths))
            self.num_fake_images = len(fake_img_paths)
        
        else:
            self.imgfilenames.extend(real_img_paths)
            self.labels.extend([0]*len(real_img_paths))
            self.num_real_images = len(real_img_paths)

            self.imgfilenames.extend(fake_img_paths)
            self.labels.extend([1]*len(fake_img_paths))
            self.num_fake_images = len(fake_img_paths)



    def __len__(self):
        return len(self.imgfilenames)


    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
        label=self.labels[idx]

        if self.transform:
            image = self.transform(image)

        tmpdir = os.path.dirname(self.imgfilenames[idx])      
        ind = tmpdir.rfind('/')
        ind = tmpdir[:ind].rfind('/')
        stub = tmpdir[ind+1:]

        fn = os.path.join(stub, os.path.basename(self.imgfilenames[idx]) )

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx],'relfilestub': fn}

        return sample



def get_dataloader(root_dir, have_classes, max_num, transform, bsize, onlyreal, onlyfake):
    """
    Get dataloader
    """
    ds = dataset_gan_vs_real(root_dir = root_dir, have_classes=have_classes, max_num=max_num, transform = transform,
            onlyreal=onlyreal, onlyfake=onlyfake)
    dl = torch.utils.data.DataLoader(ds, batch_size= bsize, shuffle=False) 

    return dl


def get_classwise_dataloader(root_dir, have_classes, max_num, transform, bsize, onlyreal, onlyfake):
    """
    Get dataloader
    """
    dls = []
    clsses = os.listdir(root_dir)

    for clss in clsses:
        clss_dir = os.path.join(root_dir, clss)
        ds = dataset_gan_vs_real(root_dir = clss_dir, have_classes=False, max_num=max_num, transform = transform,
                onlyreal=onlyreal, onlyfake=onlyfake)
        dl = torch.utils.data.DataLoader(ds, batch_size= bsize, shuffle=False)

        dls.append(dl)

    return dls







