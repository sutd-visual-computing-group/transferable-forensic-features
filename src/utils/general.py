import sys, os
import pandas as pd 

# Import torch and dependencies
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve  
import numpy as np

from termcolor import colored

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_all_channels(fake_csv_path, topk):
    """
    Get topk channels from the saved csv.
    """
    df_fake = pd.read_csv(fake_csv_path)
    df_fake = df_fake.sort_values(by="mean_relevance", ascending=False)
    all_channels = df_fake.key

    if topk == 0:
        return [], [], all_channels
    else:    
        top_fake_channels = df_fake.key.tolist()[:topk]
        # Now see relevant fake ones that are also in real ones
        topk_channels = list(top_fake_channels)
        lowk_channels = df_fake.key[-topk:]

        print("#topk channels =", len(topk_channels))
        print("#lowk channels =", len(lowk_channels))
        print("#all channels =", len(all_channels))
        return topk_channels, lowk_channels, all_channels


def get_resnet50_universal_detector(weightpath):
    """
    Get ResNet50 model loaded into the device.
    
    Args:
        weightspath : path of berkeley classifier weights

    Returns ResNet50 pytorch object
    """

    model0 = models.resnet50(pretrained=False, num_classes=1)

    somedict = torch.load(weightpath)
    model0.load_state_dict( somedict['model']   )

    # set name as attribute to each individual modules. Hooks will be attached using this name
    for n, m in model0.named_modules():
        m.auto_name = n

    return model0



def get_efb0_universal_detector(weightpath):
  """
  Get Efficient-B0 model loaded into the device.
  
  Args:
    weightspath : path of berkeley classifier weights

  Returns Efficient-B0 pytorch object
  """
  
  model0 = EfficientNet.from_name('efficientnet-b0', num_classes=1, image_size=None)
  somedict = torch.load(weightpath)
  model0.load_state_dict( somedict['model']   )
  model0.eval()

  # set name as attribute to each individual modules. Hooks will be attached using this name
  for n, m in model0.named_modules():
    m.auto_name = n
  
  return model0




def get_probs(model, dataloaders, device):
    model.eval()
    probs = []
    gt = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for index, data in enumerate(dataloader):
                imagetensors = data['image'].to(device)
                fnames = data['filename']
                relfnames = data['relfilestub']
                labels = data['label']

                prob = model(imagetensors).sigmoid().flatten().detach().cpu().numpy()
                probs.extend(list(prob))
                gt.extend(list(labels.cpu().numpy().flatten()))

    return np.asarray(probs), np.asarray(gt)


def get_calibrated_thres(y_true, y_pred, num_samples=None):
    cal_y_true = np.concatenate( [ y_true[y_true==0][:num_samples], y_true[y_true==1][:num_samples] ] )
    cal_y_pred = np.concatenate( [ y_pred[y_true==0][:num_samples], y_pred[y_true==1][:num_samples] ] )

    fpr, tpr, thresholds = roc_curve(cal_y_true, cal_y_pred)

    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))

    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = thresholds[index]
    gmeanOpt = gmean[index]
    fprOpt = fpr[index]
    tprOpt = tpr[index]
    print(colored("> Calibration results", 'cyan'))
    print('Best Threshold: {:.6f} with G-Mean: {:.6f}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    threshold = thresholdOpt

    return threshold



def get_ap_and_acc(model, device, dl, threshold):
    y_pred, y_true = get_probs(model, dl, device)

    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)

    #print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))

    return (ap, r_acc, f_acc, acc, y_pred[y_true==0].mean(), y_pred[y_true==1].mean(), y_pred[y_true==0].std(), y_pred[y_true==1].std() ), \
         (y_pred, y_true)


def get_ap_and_acc_with_new_threshold(y_pred, y_true, threshold, prefix):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)

    print('{} => AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(prefix, ap*100., acc*100., r_acc*100., f_acc*100.))

    return (ap, r_acc, f_acc, acc, y_pred[y_true==0].mean(), y_pred[y_true==1].mean(), y_pred[y_true==0].std(), y_pred[y_true==1].std() ), \
         (y_pred, y_true)



def get_ap_and_acc_random(models, device, dl, threshold, recalibrate=False):
    y_pred_global = []
    y_true = []

    for i in range(len(models)):
        y_pred, y_true = get_probs(models[i], dl, device)
        y_pred_global.append(y_pred.flatten().tolist()) # append to global
    
    y_pred = np.mean(np.asarray(y_pred_global), axis=0)

    if recalibrate:
        print("Recalibrating...")
        threshold = get_calibrated_thres(y_true, y_pred)

    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)

    #print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))

    return (ap, r_acc, f_acc, acc, y_pred[y_true==0].mean(), y_pred[y_true==1].mean(), y_pred[y_true==0].std(), y_pred[y_true==1].std() ), \
         (y_pred, y_true)