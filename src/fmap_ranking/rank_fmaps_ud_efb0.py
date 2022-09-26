# This piece of code requires high memory (RAM) for execution as relevance scores for every feature map
# is stored in memory before aggregating. 


# Import base libraries
import os, sys, math, gc
from xml.dom import NotFoundErr
import PIL
import copy
import random, json
from collections import OrderedDict

# Import scientific computing libraries
import numpy as np

# Import torch and dependencies
import torch
from torchvision import models, transforms
import torchvision


from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import load_pretrained_weights

# Import LRP modules
from utils.heatmap_helpers import *
from lrp.ef_lrp_general import *
from lrp.ef_wrapper import *

# Import utils modules
from utils.dataset_helpers import *
from utils.general import *

# Import other libraries
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def get_wrapped_efficientnet_b0(weightpath, key, device):
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

  
  model0 = EfficientNet.from_name('efficientnet-b0', num_classes=1, image_size=None,)
  somedict = torch.load(weightpath)
  model0.load_state_dict( somedict['model']   )
  model0.eval()
  
  model_e = EfficientNet_canonized.from_pretrained('efficientnet-b0', num_classes=1, image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
  
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
      probs = outputs.clone().sigmoid().flatten()
      print(probs)
      preds_labels = torch.where(probs>=0.5, 1.0, 0.0).long()
      correct_pred_indices = torch.where(torch.eq(preds_labels, label))[0]
      
      if not correct_pred_indices.size(0) > 0:
        return all_lrp_explanations
  
  #Propagate the signals for the correctly predicted samples for LRP (We should get the same LRP results if we use all samples as well.)
  with torch.enable_grad():
      if minus_fx:
        z = torch.sum( -outputs[correct_pred_indices, :] ) # Explain -f(x) if images are real
      
      else:
        z = torch.sum( outputs[correct_pred_indices, :] ) # Explain f(x) if images are fake

  with torch.no_grad():
    z.backward(retain_graph=True)
    rel = imagetensor.grad.data.clone()

    for b in range(imagetensor.shape[0]):
      # Check for correct preds and skip incorrect preds. Look for high subject GAN confidence samples
      cond = (probs[b].item() >= 0.90 and label[b].item() == 1) or (probs[b].item() <= 0.10 and label[b].item() == 0)
      
      if not cond:
        continue

      fn = relfname[b]
      lrp_explanations = {}
      lrp_explanations['relfname'] = relfname[b]
      lrp_explanations['prob'] = probs[b].item()
      
      for i, (name, mod) in enumerate(model.named_modules()):
        if hasattr(mod, 'relfromoutput'):
          v = getattr(mod, 'relfromoutput')
          
          ftrelevances = v[b,:]

          # Save feature relevances to LRP explanations dict. Move to cpu since data is big.
          lrp_explanations[name] = ftrelevances.detach().cpu()
      
      # Save LRP explanations with images as png files and also lrp explanations only as .pt files
      if save:       
        if label[b].item() == 0:
          vis_dir_name = os.path.join(outpath, "visualization", "0_real")
          vis_fname = os.path.join( vis_dir_name, fn.replace('/', '-') ).replace('.png', '.pdf')
          os.makedirs(vis_dir_name, exist_ok=True)
          
          save_img_lrp_overlay_only_positive(rel[b].to('cpu'), imagetensor[b].to('cpu'), 
              title="Label: {}, prob :{:.3f}".format( label[b].item(), probs[b].item() ), 
              q=100, outname=vis_fname )

          # Store LRP values
          lrp_dir_name = os.path.join(outpath, "lrp", "0_real")
          lrp_fname = os.path.join(lrp_dir_name, fn.replace('/', '-') + '.pt')
          os.makedirs(lrp_dir_name, exist_ok=True)
          torch.save(torch.sum(rel, dim=0).cpu(), lrp_fname) 
          
        else:
          vis_dir_name = os.path.join(outpath, "visualization", "1_fake")
          vis_fname = os.path.join( vis_dir_name, fn.replace('/', '-') ).replace('.png', '.pdf')
          os.makedirs(vis_dir_name, exist_ok=True)
          
          save_img_lrp_overlay_only_positive(rel[b].to('cpu'), imagetensor[b].to('cpu'), 
              title="Label: {}, prob :{:.3f}".format( label[b].item(), probs[b].item() ), 
              q=100, outname=vis_fname)
          
          lrp_dir_name = os.path.join(outpath, "lrp", "1_fake")
          lrp_fname = os.path.join(lrp_dir_name, fn.replace('/', '-') + '.pt')
          os.makedirs(lrp_dir_name, exist_ok=True)
          torch.save(torch.sum(rel, dim=0).cpu(), lrp_fname)

      
      all_lrp_explanations.append(lrp_explanations)
  
      torch.cuda.empty_cache()
      gc.collect()
    
    del ftrelevances

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


def normalize_relevances(lrp_explanations, only_positive=False, eps=1e-6):
  """
  Sample wise Normalize LRP explanations. We use two schemes.
  Scheme 1: Use only positive relevances for normalization
  Scheme 2 (Used in the submissiom): Use absolute relevances for normalization
  Args:
    lrp_explanations  : Relevance values
    only_positive     : If set to True, use scheme 1. If False, use scheme 2
    eps               : An epsilon value to avoid division by zero 
  Returns:
    Sample wise normalized lrp explanations
  """
  layer_names = lrp_explanations[0].keys()
  assert only_positive == False

  for layer_name in layer_names:
    if layer_name in ['relfname', 'prob', 'f(x)', 'y']:
      continue
    
    for i in range(len(lrp_explanations)):
      if only_positive:
        normalization_constant = torch.sum( torch.maximum( torch.tensor([0.0]), lrp_explanations[i][layer_name] ) ) 

        if torch.allclose(normalization_constant, torch.tensor([0.0]), atol=1e-8):
          lrp_explanations[i][layer_name] = torch.zeros(size=lrp_explanations[i][layer_name].size())
          
        else:
          lrp_explanations[i][layer_name] = torch.div( torch.maximum( torch.tensor([0.0]), lrp_explanations[i][layer_name] ), 
                                        normalization_constant )
          assert torch.allclose( torch.sum(lrp_explanations[i][layer_name]), torch.tensor([1.0]) ), "{}:{}".format(normalization_constant.item(), layer_name)

      
      else:
        normalization_constant = torch.sum( torch.abs( lrp_explanations[i][layer_name]) ) # Total amount of "evidence"
        #print(layer_name, lrp_explanations[i][layer_name].size(), normalization_constant.item(), torch.sum( lrp_explanations[i][layer_name] ).item())

        # Now look at the ratio of positive evidence given the total absolute evidence.
        lrp_explanations[i][layer_name] = torch.div( torch.maximum( torch.tensor([0.0]), lrp_explanations[i][layer_name] ), 
                                        normalization_constant )
        
  return lrp_explanations


def calculate_channelwise_stats(normalized_lrp_explanations):
  """
  This is the pipeline to samplewise normalize the relevances and obtain the topk discriminative channels

  Args:
    normalized_lrp_explanations : Take \hat{R} and return the channelwsie stats
  """

  layer_names = normalized_lrp_explanations[0].keys()
  channelwise_stats = {}

  for layer_name in layer_names:
    if layer_name in ['relfname', 'prob']:
      continue

    list_norm = [ normalized_lrp_explanations[i][layer_name] for i in range(len(normalized_lrp_explanations)) ]
    stack_norm = torch.stack(list_norm)
    
    average_fmap_relevances = torch.sum( torch.mean(stack_norm, dim=0), dim=(1, 2)) # We obtain the averaged feauture map, then sum over h, w

    for channel_idx in range(average_fmap_relevances.shape[0]):
      channelwise_stats["{}.#{}(T={})".format(layer_name, channel_idx, average_fmap_relevances.shape[0])] = average_fmap_relevances[channel_idx]

  return channelwise_stats


def pipeline(model_e, dl, device, outpath, save, 
            num_instances,
            minus_fx, normalize_only_using_positive, topk):
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
  blockPrint()
  all_lrp_explanations = get_all_lrp_positive_explanations(model_e, dl, device, outpath, save, minus_fx)
  enablePrint()

  if len(all_lrp_explanations) == 0:
    print("issue here")
    return None, None

  # Normalize
  norm_lrp = normalize_relevances(all_lrp_explanations, only_positive=normalize_only_using_positive)

  # Calculate channelwise stats
  channelwise_stats = calculate_channelwise_stats(norm_lrp)

  # Calulate top k channels
  channelwise_stats = sorted(channelwise_stats.items(), key=lambda x: x[1], reverse=True)
  topk_channelwise_stats = channelwise_stats[:topk]

  final_channelwise_stats = OrderedDict()
  for i in range(len(topk_channelwise_stats)):
    key, value = topk_channelwise_stats[i][0], topk_channelwise_stats[i][1]
    final_channelwise_stats[key] = value

  #print(final_channelwise_stats)
  return final_channelwise_stats, len(all_lrp_explanations)



def save_results_as_csv(channelwise_stats, key1, key2, save_name):
  """
  Save as CSV. Preliminary Analysis can be done first using excel before automation

  Args:
    channelwise_stats : dict of stats
    key1              : name of layers
    key2              : Real or fake
  """
  df = pd.DataFrame(columns=[key1, key2])
  df[key1] = channelwise_stats.keys()
  df[key2] = [i.item() for i in channelwise_stats.values()]
  df.to_csv('{}.csv'.format(save_name), index=False)
  return df


def rank_fmaps(parent_dir, gan_name, 
        aug, have_classes, 
        bsize,
        num_instances_real, num_instances_fake,
        save_pt_files,
        normalize_only_using_positive = False,
        topk=None):
  ## ------------Set Parameters--------
  # Define device and other parameters
  device = torch.device('cuda:0')

  # LRP model keys and weights
  key = 'beta0' # 'beta0' 'beta1' , 'betaada'
  weightfn = './weights/efb0/blur_jpg_prob{}.pth'.format(aug)

  # Directories
  root_dir = os.path.join(parent_dir, gan_name)
  outpath_channelwise = os.path.join('./fmap_relevances/', 'efb0', '{}_{}'.format(gan_name, aug)) 
  outpath = './output/efb0/{}_{}/'.format(gan_name, aug) # For saving .pt files if required

  ## ------------End of Parameters--------

  # Model
  blockPrint()
  model_e = get_wrapped_efficientnet_b0(weightfn, key, device)
  enablePrint()
  print("> LRP wrapped ResNet-50 loaded successfully")
  
  def writeintomodule_bwhook(self,grad_input, grad_output):
    #gradoutput is what arrives from above, shape id eq to output
    setattr(self,'relfromoutput', grad_output[0])

  # Register hook
  for i, (name,mod) in enumerate(model_e.named_modules()):
    #print(i,name )
    #  if (('conv' in name) and ('module' not in name)) or (name in ['layer1', 'layer2', 'layer3', 'layer4'])
    
    if ((('conv' in name) or ('downsample.0' in name)) and ('module' not in name)):
      # print(name, 'ok')
      mod.register_full_backward_hook(writeintomodule_bwhook)   # modify to full_backward_hook

  print("> All backward hooks registered successfully")

  # Dataset (Use a Random Resized Crop so that you don't ignore any boundary artifacts on images)
  # Do note you might expect very small numerical changes when you repeat this experiments due to random resized crop
  # But the resulting ranking and topk feature maps should be the same.
  
  transform = transforms.Compose([
          torchvision.transforms.RandomResizedCrop(224, 
                scale=(0.99, 1.0), ratio=(0.99, 1.00), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  
  # Obtain D_real and D_fake
  dl_real = get_dataloader(root_dir, have_classes, num_instances_real , transform, bsize, onlyreal=True, onlyfake=False)
  dl_fake = get_dataloader(root_dir, have_classes, num_instances_fake , transform, bsize, onlyreal=False, onlyfake=True)
  print("> Dataloaders loaded successfully")

  # Pass to the overall algorithm pipeline to obtain C_real_topk, C_fake_topk
  print("> Calculating feature map relevances (This will take some time)")
  real_channelwise_stats, num_real_samples = pipeline(model_e, dl_real, device, outpath, save=save_pt_files, 
                                    num_instances=num_instances_real,
                                    minus_fx=True, normalize_only_using_positive=normalize_only_using_positive, topk=topk)
  model_e.eval()
  fake_channelwise_stats, num_fake_samples = pipeline(model_e, dl_fake, device, outpath, save=save_pt_files, 
                                    num_instances=num_instances_fake,
                                    minus_fx=False, normalize_only_using_positive=normalize_only_using_positive, topk=topk)

  # Save as csv for seperate analysis
  print("> Ranking feature map relevances")
  outpath_channelwise = os.path.join(outpath_channelwise)
  os.makedirs(outpath_channelwise, exist_ok=True)
  _ = save_results_as_csv(real_channelwise_stats, 'key', 'mean_relevance', os.path.join(outpath_channelwise, 
                        '{}-real'.format(gan_name)) )
  _ = save_results_as_csv(fake_channelwise_stats, 'key', 'mean_relevance',  os.path.join(outpath_channelwise, 
                        '{}-fake'.format(gan_name)) )
  
  print( "> Completed..." )
  print( "> #real used = {}, #fake used = {}".format(num_real_samples, num_fake_samples) )

  return


if __name__=='__main__':
  rank_fmaps()
