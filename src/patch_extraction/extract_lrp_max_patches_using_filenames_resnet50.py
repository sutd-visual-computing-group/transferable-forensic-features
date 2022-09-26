

# from utils import mask_topk_channels
import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt

from torch import Tensor

import datetime
import time
import os
import numpy as np

import PIL.Image


#LRP

from lrp.resnet_wrapper import *
from utils.heatmap_helpers import *
from utils.dataset_helpers import *
from utils.general import *
from utils.mask_fmaps import *

from utils import *


# Keep penultimate features as global varialble such that hook modifies these features
penultimate_fts = None


def get_penultimate_fts(self, input, output):
    global penultimate_fts
    penultimate_fts = output
    return None


class Dataset_from_paths(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, transform=None):
        self.transform = transform
        self.imgfilenames=img_paths

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
        label=1

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx] }

        return sample


def interpret_simtest(model, actual_model,  dataloader, probs, 
                    device, savept, 
                    layer_name, channel_index,
                    bias_inducer):
    #print(probs)
    if not os.path.isdir(savept):
      os.makedirs(savept)

    model.eval()
    fnames = []
    rank = 0

    max_act_values = []
    corr_filenames = []

    for batch_idx, data in enumerate(dataloader):

        if (batch_idx%100==0) and (batch_idx>=100):
          print('at val batchindex: ',batch_idx)
    
        inputs = data['image'].to(device)        
        labels = data['label']          
        if bias_inducer is not None:
          inputs=bias_inducer(inputs, labels)
          
        fnames.extend(data['filename'])
        inputs.requires_grad=True
        print('inputs.requires_grad',inputs.requires_grad)

        with torch.no_grad():
          global penultimate_fts
          penultimate_fts = None
          assert(penultimate_fts == None)
          actual_activation = actual_model(inputs)
          #sys.exit()
          assert torch.is_tensor(penultimate_fts)

        with torch.enable_grad():
          outputs = model(inputs)

        for b in range(outputs.shape[0]): 
          rank += 1
          # Look for high activating samples
          maxpool_activation = torch.max(penultimate_fts[:, channel_index, :, :])

          #print('inp',inputs.shape, outputs.shape)    
          if inputs.grad is not None:
            inputs.grad.zero_() 
          
          prez = outputs[b]
          mask = torch.zeros_like(prez)   
          mask[outputs[b] > 0] = 1.0
          
          # Create prez
          print("prez size", prez.size())
          prez2= prez*mask
          
          #find maxind
          vh,indh = torch.max(prez2,dim=0)
          vw,indw = torch.max(vh,dim=0)
          z=vh[indw]

          #find sum
          #print("prez size", prez2.size(), outputs.size())
          #z=torch.sum(prez2)
          
          print('z val', z.item())

        #   if (labels[b].item() == 0):
        #     z = -z

          with torch.no_grad(): 
            z.backward()
            rel=inputs.grad.data.clone().detach().cpu()      
            # rel= rel/torch.abs(torch.sum(rel))
            #if outputs[b,n]<thresh[n] :
            #  rel*=-1.

            # if (labels[b].item() == 0):
            #   rel *= -1
                                
            fn = data['filename'][b]
            os.makedirs(os.path.join(savept, "visualization"), exist_ok=True)
            # savenm = os.path.join(savept, "visualization", os.path.basename(fn)+'{}_hm_{:.3f}_{:.3f}.jpg'.format(maxpool_activation.item(), z.item()) )

            savenm = os.path.join(savept, "visualization", '{}_hm_{:.3f}_{:.3f}.jpg'.format(rank, maxpool_activation.item(), z.item()) )
            print(savenm, rel[b].size(), inputs[b].size())

            extract_patch( rel, inputs.cpu(), 
                  title=os.path.basename(fn)+" | p={:.3f} | {}#{} | {:.3f}".format(probs[batch_idx], layer_name, channel_index, maxpool_activation.item()), 
                    q=100, outname=savenm)

            max_act_values.append(maxpool_activation.item())
            corr_filenames.append(os.path.basename(fn))
          

    return max_act_values, corr_filenames


#***************************


class ResNet_canonized_modfwd(ResNet_canonized):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_canonized_modfwd, self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation,
                 norm_layer)

      
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        rettensor=None
        for ind,(name,module) in enumerate(self.named_modules()):
          if hasattr(module,'tempstorefeature'):
            rettensor = getattr(module,'tempstorefeature').clone()
            print('found tempstorefeature at',ind, name)
            #clean up chicken shit
            delattr(module,'tempstorefeature')
            
        if rettensor is not None:
          return rettensor
          
        print('no special feature map found')
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)      
      
      
def _resnet_canonized_modfwd(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_canonized_modfwd(block, layers, **kwargs)
    if pretrained:
        raise Cannotloadmodelweightserror("explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model")
    return model


def resnet18_canonized_modfwd(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized_modfwd('resnet18', BasicBlock_fused, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet50_canonized_modfwd(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized_modfwd('resnet50', Bottleneck_fused, [3, 4, 6, 3], pretrained, progress, **kwargs)




def onlywritefeaturemap_hook(module, input_ , output, channelind):

  if channelind is None:
    module.tempstorefeature=output
  else:
    module.tempstorefeature=output[:,channelind]
    
def hook_factory2(channelind):

    # define the function with the right signature to be created
    def ahook(module, input_, output):
        # instantiate it by taking a parametrized function,
        # and fill the parameters
        # return the filled function
        return onlywritefeaturemap_hook(module, input_, output,  channelind =  channelind)

    # return the hook function as if it were a string
    return ahook

  

def get_probs(model, data_loader, device):
  y_true, y_pred = [], []
  Hs, Ws = [], []
  
  from tqdm import tqdm 

  with torch.no_grad():
      for datas in tqdm(data_loader):
        #print(datas['label'].size())
        data = datas['image'].to(device)
        label = datas['label']      
        # for data, label in data_loader:
        Hs.append(data.shape[2])
        Ws.append(data.shape[3])

        y_true.extend(label.flatten().tolist())
        data = data.cuda()
        y_pred.extend(model(data).sigmoid().flatten().tolist())

  Hs, Ws = np.array(Hs), np.array(Ws)
  y_true, y_pred = np.array(y_true), np.array(y_pred).astype(np.float16)

  print(np.count_nonzero(y_pred[y_true==1]>=0.5))

  return y_pred


def get_high_activation_patches(feature_map_name, arch, gan_name, aug, bsize, num_instances=None):
    ## ------------Set Parameters--------
    # Define device and other parameters
    device = torch.device('cuda:0')
    key = 'beta0'

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

    # Use original transform as Wang et. al
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    transform_crop = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Read csv
    df = pd.read_csv('output/activation_rankings/{}/{}_{}/{}.csv'.format(arch, gan_name, aug, feature_map_name))
    img_paths = df['name'][:num_instances]
    ds = Dataset_from_paths(img_paths, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size= 1, shuffle=False) 

    ds_ = Dataset_from_paths(img_paths, transform=transform_crop)
    dl_prob = torch.utils.data.DataLoader(ds_, batch_size= bsize, shuffle=False) 
    
    # Load LRP wrapped model
    #model = get_resnet50_universal_detector(weightfn).to(device)

    lrp_params_def1={
      'conv2d_ignorebias': True, 
      'eltwise_eps': 1e-6,
      'linear_eps': 1e-6,
      'pooling_eps': 1e-6,
      'use_zbeta': True ,
      }

    lrp_layer2method={
    'nn.ReLU':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    'nn.Conv2d':        conv2d_beta0_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'nn.MaxPool2d': maxpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
    }

    model_e =  resnet50_canonized_modfwd(pretrained=False)
    model_e.copyfromresnet(model, lrp_params=lrp_params_def1, lrp_layer2method = lrp_layer2method)
    model_e = model_e.to(device)
    
    model.eval()
    model_e.eval()

    # Get output probs
    preds = get_probs(model, dl_prob, device)

    # ---------------------------------------------
    for n, m in model.named_modules():
          m.auto_name = n
    
    for n, m in model_e.named_modules():
          m.auto_name = n
    
    # Attach hooks to LRP model
    handles=[]
    for ind,(name,module) in enumerate(model_e.named_modules()):
        print('name: {}'.format(name) )
        if name ==layertobeattached:
          h=module.register_forward_hook( hook_factory2( channelind = feature_map_idx ))
          handles.append(h)

    # Attach hook to original model
    new_handles = []
    for ind,(name,module) in enumerate(model.named_modules()):
        if name ==layertobeattached:
          print('name: {}'.format(name) )
          h=module.register_forward_hook( get_penultimate_fts )
          new_handles.append(h)

    #sys.exit()

    # do this only for a subset by subsetting the dataloader
    # fwd hook copies module, return check through if a module has a feature
    save_suffix = 'fake'
    save_dir = os.path.join('./output/patches/', arch, gan_name, save_suffix, "{}.#{}".format(layertobeattached, feature_map_idx))
    max_act_vals, corr_filenames = interpret_simtest(model_e, model, dataloader = dl,  probs=preds, device = device, 
                              savept='{}/'.format(save_dir),
                              layer_name= layertobeattached, channel_index=feature_map_idx,  bias_inducer = None )
    
    max_act_vals = np.asarray(max_act_vals)
    enablePrint()


# if __name__=='__main__':

#     # Read fmaps
#     df = pd.read_csv("progan_val-grayscale.csv")
#     df = df[df['p_value'] < 0.05]
#     print(df.shape)
#     print(df)
#     fmaps = list(df['feature_map_name'])[:]

#     # fmaps = [
#     #  '"layer4.2.conv1.#487(T=512)"',

#     # ]
#     for i in range(len(fmaps)):
#         feature_map_name  = fmaps[i]
#         print(i, feature_map_name)
#         extract_patches( feature_map_name, None)
