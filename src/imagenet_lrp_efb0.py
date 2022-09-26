# Import generic libraries
import os, sys, math, gc
import PIL

# Import scientific computing libraries
import numpy as np

# Import torch and dependencies
import torch
from torchvision import models, transforms

# Import other libraries
import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import load_pretrained_weights

# Import LRP modules
from utils.heatmap_helpers import *
from lrp.ef_lrp_general import *
from lrp.ef_wrapper import *

# Import utils
from utils.heatmap_helpers import *
from utils.dataset_helpers import *
from utils import *

# Import other modules
import copy
from collections import OrderedDict
import pandas as pd


import argparse
parser = argparse.ArgumentParser(description='Rank feature maps of universal detectors...')

# architecture
parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'efb0'])

# architecture
parser.add_argument('--classifier', type=str, required=True, choices=['imagenet', 'ud'])

args = parser.parse_args()


def get_wrapped_efficientnet_b0(weightpath, key, device, classifier):
  """
  Get Wrapped ResNet50 model loaded into the device.
  
  Args:
    weightspath : path of berkeley classifier weights
    key         : LRP key
    device      : cuda or cpu to store the model

  Returns resnet50 pytorch object
  """

  if key == 'beta0':
    #beta0
    lrp_params_def1={
    'conv2d_ignorebias': True, 
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': False ,
    }

    lrp_layer2method={
    'Swish':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    'nn.Conv2d':         Conv2dDynamicSamePadding_beta0_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
    }
    
  elif  key == 'beta1': 
     pass
  elif  key == 'betaada': 
     pass
  else:
      raise NotImplementedError("Unknown key", key)

  if classifier == 'ud':
  
    model0 = EfficientNet.from_name('efficientnet-b0', num_classes=1, image_size=None,)
    somedict = torch.load(weightpath)
    model0.load_state_dict( somedict['model']   )
    model0.eval()
    #load_pretrained_weights(model0, 'efficientnet-b0', weightpath)
    #model0.set_swish( memory_efficient=False)
    
    model_e = EfficientNet_canonized.from_pretrained('efficientnet-b0', num_classes=1, image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
    #print(model_e)
    #model_e.set_swish( memory_efficient=False)
    
    model_e.copyfromefficientnet( model0, lrp_params_def1, lrp_layer2method)
    model_e.to(device)

    return model_e

  else:
    model0 = EfficientNet.from_pretrained('efficientnet-b0', image_size=None)
    model0.eval()
    #load_pretrained_weights(model0, 'efficientnet-b0', weightpath)
    #model0.set_swish( memory_efficient=False)
    
    model_e = EfficientNet_canonized.from_pretrained('efficientnet-b0', num_classes=1000, image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
    #print(model_e)
    #model_e.set_swish( memory_efficient=False)
    
    model_e.copyfromefficientnet( model0, lrp_params_def1, lrp_layer2method)
    model_e.to(device)

    return model_e




def get_lrp_explanations_for_batch(model, 
                                  imagetensor, label, 
                                  relfname , save, outpath, minus_fx=False): 
  """
  Get LRP explanations for a single sample.

  Args:
    model         : pytorch model
    imagetensor   : images
    label         : label
    relfname      : filenames
    save          : If True, save LRP explanations
    outpath       : output path to save pixel space explanations

  Get dict with positive relevances for the sample
  """
  model.eval()

  all_lrp_explanations = []

  os.makedirs(outpath, exist_ok=True)

  if imagetensor.grad is not None:
      imagetensor.grad.zero_() 
    
  imagetensor.requires_grad=True # gxinp needs it here
  
  with torch.enable_grad():
      outputs = model(imagetensor)

  with torch.no_grad():
      # probs = outputs.sigmoid().flatten()
      # preds_labels = torch.where(probs>0.5, 1.0, 0.0).long()
      # correct_pred_indices = torch.where(torch.eq(preds_labels, label))[0]
      #print(correct_pred_indices, correct_pred_indices.size(0))
      #print(outputs, outputs[correct_pred_indices, :])
      
      # if not correct_pred_indices.size(0) > 0:
      #   return all_lrp_explanations


      # For imagenet:
      probs = outputs.softmax(dim=1).flatten()
      #preds_labels = torch.where(probs>0.5, 1.0, 0.0).long()
      #correct_pred_indices = torch.where(torch.eq(preds_labels, label))[0]
      _, predclasses = torch.max(outputs, 1)
  
  #Propagate the signals for the correctly predicted samples for LRP (We should get the same LRP results if we use all samples as well.)
  # with torch.enable_grad():
  #     if minus_fx:
  #       z = torch.sum( -outputs[correct_pred_indices, :] ) # Explain -f(x) if images are real
      
  #     else:
  #       z = torch.sum( outputs[correct_pred_indices, :] ) # Explain f(x) if images are fake

  # For imagenet
  with torch.enable_grad():
    if minus_fx:
        z = torch.sum( -outputs[:, predclasses] ) # Explain imagenet
        
    else:
        z = torch.sum( outputs[:, predclasses] ) # Explain imagenet
      
  with torch.no_grad():
    z.backward(retain_graph=True)
    rel = imagetensor.grad.data.clone()

    for b in range(imagetensor.shape[0]):
      # Check for correct preds and skip incorrect preds
      # cond = (probs[b].item() >= 0.5 and label[b].item() == 1) or (probs[b].item() < 0.5 and label[b].item() == 0)
      
      # if not cond:
      #   continue

      fn = relfname[b]
      lrp_explanations = {}
      lrp_explanations['relfname'] = relfname[b]
      lrp_explanations['prob'] = probs[b].item()
      
      for i, (name, mod) in enumerate(model.named_modules()):
        if hasattr(mod, 'relfromoutput'):
          v = getattr(mod, 'relfromoutput')
          #print(i, name, v.shape) # conv rel map
          
          ftrelevances = v[b,:]
          
          # take only positives
          #ftrelevances[ftrelevances<0] = 0

          # Save feature relevances to LRP explanations dict. Move to cpu since data is big.
          #lrp_explanations[name] = ftrelevances.detach().cpu()

      # All LRP explanations       
      if label[b].item() == 0:
        vis_dir_name = os.path.join(outpath, "visualization", "0_real")
        vis_fname = os.path.join( vis_dir_name, fn.replace('/', '-').replace('.png', '-p={:.3f}.pdf'.format(probs[b].item())) )
        vis_fname = vis_fname.replace('.JPEG', '.pdf')
        print(vis_fname)
        os.makedirs(vis_dir_name, exist_ok=True)
        
        save_img_lrp_overlay_only_positive(rel[b].to('cpu'), imagetensor[b].to('cpu'), 
            title="Label: {}, prob :{:.3f}".format( label[b].item(), probs[b].item() ), 
            q=100, outname=vis_fname )

        # Store LRP values
        if save:
          lrp_dir_name = os.path.join(outpath, "lrp", "0_real")
          lrp_fname = os.path.join(lrp_dir_name, fn.replace('/', '-') + '.pt')
          os.makedirs(lrp_dir_name, exist_ok=True)
          torch.save(torch.sum(rel[b], dim=0).cpu(), lrp_fname) 
        
      else:
        vis_dir_name = os.path.join(outpath, "visualization", "1_fake")
        vis_fname = os.path.join( vis_dir_name, fn.replace('/', '-').replace('.png', '-p={:.3f}.pdf'.format(probs[b].item())) )
        vis_fname = vis_fname.replace('.JPEG', '.pdf')
        os.makedirs(vis_dir_name, exist_ok=True)
        
        save_img_lrp_overlay_only_positive(rel[b].to('cpu'), imagetensor[b].to('cpu'), 
            title="Label: {}, prob :{:.3f}".format( label[b].item(), probs[b].item() ), 
            q=100, outname=vis_fname)

        if save:
          lrp_dir_name = os.path.join(outpath, "lrp", "1_fake")
          lrp_fname = os.path.join(lrp_dir_name, fn.replace('/', '-') + '.pt')
          os.makedirs(lrp_dir_name, exist_ok=True)
          torch.save(torch.sum(rel[b], dim=0).cpu(), lrp_fname)

      #all_lrp_explanations.append(lrp_explanations)
  
      torch.cuda.empty_cache()
      gc.collect()
    
    # del ftrelevances

  return all_lrp_explanations


def get_all_lrp_positive_explanations(model, dataloader, device, outpath, save, minus_fx):
  """
  Get all LRP explanations for one folder. 

  Args:
    model       : resnet50 pytorch model
    dataloader  : pytorch dataloader 
    device      : 
    outpath     : output path to save visualization and lrp numpy files
    save        : If set to True, save
    minus_fx    : If set to True, we will use -f(x) signal to calculate relevances

  Returns all LRP explanations
  """
  # Global variable to store feature map information
  all_lrp_explanations = []

  # > Explain prediction
  for index, data in enumerate(dataloader):
    # Get image tensor, filename, stub and labels
    imagetensors = data['image'].to(device)
    fnames = data['filename']
    relfnames = data['relfilestub']
    labels = data['label'].to(device)
    
    # Get LRP explanations
    lrp_explanations = get_lrp_explanations_for_batch(model, imagetensors, labels, relfnames, save, outpath, minus_fx)

    # Get LRP heatmap for all layers
    all_lrp_explanations.extend(lrp_explanations)

    torch.cuda.empty_cache()
    gc.collect()
    del lrp_explanations

  return all_lrp_explanations



def pipeline(model_e, dl, device, outpath, save, 
            num_instances,
            minus_fx):
  """
  Pipeline to run overall algorithm for real or fake images

  Args:
    model_e : Wrapped resnet50
    dl      : dataloader
    device  : cuda/ cpu
    outpath : output path to save the channelwise stats
    save    : If set to True, all LRP relevances are saved as .pt files
    minus_fx: Needs to be set to True for real images
    normalize_using_only_positive : If set to True, scheme 1 else scheme 2.
    topk    : #topk feature maps to return.
  """
  # Get LRP explanations
  all_lrp_explanations = get_all_lrp_positive_explanations(model_e, dl, device, outpath, save, minus_fx)[:num_instances]

  #print(final_channelwise_stats)
  return all_lrp_explanations



def main():
  ## ------------Set Parameters--------
  # Define device and other parameters
  device = torch.device('cuda:0')
  bsize = 16

  # LRP model keys and weights
  key = 'beta0' # 'beta0' 'beta1' , 'betaada'
  weightfn = './weights/{}/blur_jpg_prob0.5.pth'.format(args.arch)

  # Directories
  #parent_dir = '/mnt/workspace/projects/deepfake_classifiers_interpretability/samples/'
  parent_dir = '/mnt/data/CNN_synth_testset/' # Use our version
  have_classes = False
  gan_and_classes = {}
  # gan_and_classes['biggan'] = ['beer_bottle', 'monitor', 'vase', 'table_lamp',
  #               'hummingbird', 'church', 'egyptian_cat', 'welsh_springer_spaniel']
  
  #gan_and_classes['progan_test'] = os.listdir(os.path.join(parent_dir, 'progan_test'))
  #gan_and_classes['progan_test'] = [ 'boat' ]
  #gan_and_classes['biggan'] = ['bird']
  gan_and_classes['stylegan2'] = ['church', 'car', 'cat']
  #gan_and_classes['stylegan'] = ['car']
  #gan_and_classes['cyclegan'] = ['horse']
  #gan_and_classes['stargan'] = ['person']
  #gan_and_classes['gaugan'] = ['mscoco']
  
  #gan_and_classes['san'] = ['']
  #gan_and_classes['stylegan2'] = ['car', 'cat', 'horse', 'church']
  #gan_and_classes['cyclegan'] = ['horse']
  #gan_and_classes['stargan'] = ['']
  
  
  for gan_name in gan_and_classes:
    for clss in gan_and_classes[gan_name]:
      print(gan_name, clss)
      root_dir = os.path.join(parent_dir, gan_name, clss)
      #clss='sr'
      outpath = './hms/lrp_heatmaps_{}_{}/{}/{}/'.format(args.arch, args.classifier, gan_name, clss)
      save_pt_files = False # No need to save .pt files.
      num_instances_real, num_instances_fake = 1, 500 # Use 1000 real and fake samples for analysis
      ## ------------End of Parameters--------


      # Model
      model_e = get_wrapped_efficientnet_b0(weightfn, key, device, args.classifier)
      
      def writeintomodule_bwhook(self,grad_input, grad_output):
        #gradoutput is what arrives from above, shape id eq to output
        setattr(self,'relfromoutput', grad_output[0])

      # Register hook
      for i, (name,mod) in enumerate(model_e.named_modules()):
        #print(i,nm)
        if ('conv' in name) and ('module' not in name):
          #print('ok')
          mod.register_backward_hook(writeintomodule_bwhook)  

      # Dataset (Use same transforms as Wang et. al without Center cropping)
      transform = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
      
      # Obtain D_real and D_fake
      dl_real = get_dataloader(root_dir, have_classes, num_instances_real , transform, bsize, onlyreal=True, onlyfake=False)
      dl_fake = get_dataloader(root_dir, have_classes, num_instances_fake , transform, bsize, onlyreal=False, onlyfake=True)

      # Pass to the overall algorithm pipeline to obtain C_real_topk, C_fake_topk
      real_lrp_explanations = pipeline(model_e, dl_real, device, outpath, save=save_pt_files, 
                                        num_instances=num_instances_real,
                                        minus_fx=True)
      fake_lrp_explanations  = pipeline(model_e, dl_fake, device, outpath, save=save_pt_files, 
                                        num_instances=num_instances_fake,
                                        minus_fx=False)



  
  return real_lrp_explanations, fake_lrp_explanations


if __name__=='__main__':
  main()
