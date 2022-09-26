import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import torch, os
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from utils.heatmap_helpers import *

from efficientnet_pytorch import EfficientNet


import argparse
parser = argparse.ArgumentParser(description='Rank feature maps of universal detectors...')

# architecture
parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'efb0'])

# architecture
parser.add_argument('--classifier', type=str, required=True, choices=['imagenet', 'ud'])

args = parser.parse_args()

def load_img_as_tensor(path):
    pil_image = Image.open(path).convert("RGB")

    pil_transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          #transforms.ToTensor(),
          #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])

    ud_transforms = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    return pil_transform(pil_image), ud_transforms(pil_image).unsqueeze(0)


def get_ud_model_r50(path):
    model = models.resnet50(pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    return model


def get_imagenet_model_r50():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


def get_ud_model_efb0(path):
    model0 = EfficientNet.from_name('efficientnet-b0', num_classes=1, image_size=None)
    somedict = torch.load(path)
    model0.load_state_dict( somedict['model']   )
    model0.eval()
    return model0


def get_imagenet_model_efb0():
    model0 = EfficientNet.from_pretrained('efficientnet-b0', image_size=None)
    model0.eval()
    return model0


def center_crop(image, h=224, w=224):
    center = [ image.shape[0] // 2, image.shape[1] // 2 ]
    #print(center)
    x = int(center[1] - w/2)
    y = int(center[0] - h/2)
    #print(x, y)
    return image[y:y+h, x:x+w]


def my_deprocess_image(img, savename):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """

    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = np.clip(img, 0, None) # consider only positive values


    #red_channel_sum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Get the grayscale values
    red_channel_sum = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])
    img[:, :, 0] = red_channel_sum/np.max(red_channel_sum) # Copy the grayscale value to red channel
    img[:, :, 1] = np.zeros((224, 224))  # Only red color is allowed
    img[:, :, 2] = np.zeros((224, 224)) # Only red color is allowed
    img = np.clip(img, 0, 1)

    cmap = plt.cm.seismic
    plt.imsave("{}".format(savename), img[:, :, 0], cmap=cmap, vmin=-1.0, vmax=1.0, format='pdf')


def get_heatmaps(
            image_path, 
            rgb_img,
            input_tensor,
            prob,
            model,
            method, 
            use_cuda, 
            gan_name,
            type,
            clss,
            arch,
            classifier):
    
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}


    if arch == 'resnet50':
        target_layers =  [model.layer4]
    else:
        target_layers = [model._conv_head]

    #print(target_layers)
    rgb_img = np.float32(rgb_img.copy()) / 255


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None
    #targets = [ClassifierOutputTarget(218)] # sorrel
    #targets = [ClassifierOutputTarget(625)] # lifeboat

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=True) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            # aug_smooth=args.aug_smooth,
                            # eigen_smooth=args.eigen_smooth
                            )

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False, colormap=cv2.COLORMAP_OCEAN)

        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    img_name = image_path.split('/')[-1].split(".")[0] + "-p={:.3f}".format(prob)
    save_dir = os.path.join("./output/hms/gradcam_heatmaps_{}_{}".format(classifier, arch ), gan_name, clss, type, img_name, method)
    os.makedirs(save_dir, exist_ok=True)

    plt.imsave(os.path.join(save_dir, "image.pdf"), rgb_img, format='pdf')

    plt.imsave(os.path.join(save_dir, "cam.pdf"), cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB), format='pdf')

    plt.imsave(os.path.join(save_dir, "gb.pdf"), cv2.cvtColor(gb, cv2.COLOR_BGR2RGB), format='pdf')

    plt.imsave(os.path.join(save_dir, "cam_gb.pdf"), cv2.cvtColor(cam_gb, cv2.COLOR_BGR2RGB), format='pdf')

    my_deprocess_image(cam_mask * gb, os.path.join(save_dir, "cam_gb_paper.pdf") )

    # My heatmap
    # Combine rgb_image + (CAM * GB)
    gb_heatmap = cam_mask * gb
    rgb_image = rgb_img

    save_img_guided_gradcam_overlay_only_positive( gb_heatmap,  rgb_image,  q=100, title=None, outname=os.path.join(save_dir, "cam_gb_paper_final.pdf"))




def main():
    ## ------------Set Parameters--------
    # Define device and other parameters
    device = torch.device('cuda:0')
    method = "gradcam"
    use_cuda=True

    # Directories
    parent_dir = '/mnt/data/CNN_synth_testset/' # Use our version
    gan_and_classes = {}
    
    # GAN names and classes (You can define the GAN and the corresponding classes)
    #gan_and_classes['progan_test'] = os.listdir(os.path.join(parent_dir, 'progan_test'))
    #gan_and_classes['progan_test'] = ['boat']
    #gan_and_classes['biggan'] = ['bird']
    gan_and_classes['stylegan2'] = ['church', 'car', 'cat']
    #gan_and_classes['stylegan'] = ['car']
    #gan_and_classes['cyclegan'] = ['horse']
    #gan_and_classes['stargan'] = ['person']
    #gan_and_classes['gaugan'] = ['mscoco']
    

    # model and weights
    if args.arch == 'resnet50':
        if args.classifier == 'ud':
            weightfn = './weights/resnet50/blur_jpg_prob{}.pth'.format(0.5)
            model = get_ud_model_r50(weightfn).to(device)
        else:
            model = get_imagenet_model_r50().to(device)
    
    elif args.arch == 'efb0':
        if args.classifier == 'ud':
            weightfn = './weights/efb0/blur_jpg_prob{}.pth'.format(0.5)
            model = get_ud_model_efb0(weightfn).to(device)
        else:
            model = get_imagenet_model_efb0().to(device)
        
        # print(model)
    

    for gan_name in gan_and_classes:
        for clss in gan_and_classes[gan_name]:
            for type in ['1_fake']:
                print(gan_name, clss)
                root_dir = os.path.join(parent_dir, gan_name, clss, type)

                sample_paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]
                sample_paths.sort()
                
                for img_path in sample_paths[:500]:
                    rgb_img, input_tensor = load_img_as_tensor(img_path)
                    prob = (model(input_tensor.to(device))).sigmoid().item()
                    #prob = torch.max(model(input_tensor.to(device))).item()
                

                    get_heatmaps(
                    img_path, 
                    rgb_img,
                    input_tensor,
                    prob,
                    model,
                    method, use_cuda,
                    gan_name, type, clss,
                    args.arch, args.classifier)



if __name__=='__main__':
    main()