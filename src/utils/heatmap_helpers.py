import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2


from colour import Color
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

from PIL import Image

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.xmargin'] = 0

def make_plot_custom_cmap( cmap_colors ):
  """
  Pass your own colors to create a custom colorbar

  params:
    cmap_colors : List of hex colors
  """
  color_cmap = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in cmap_colors ] )
  plt.figure( figsize = (15,3))
  plt.imshow( [list(np.arange(0, len( cmap_colors ) , 0.1)) ] , interpolation='nearest', origin='lower', cmap= color_cmap )
  plt.xticks([])
  plt.yticks([])
  plt.show()
  return color_cmap



def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  """
  Invert normalized images
  """
  s = torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
  m = torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

  res = ten*s+m
  return res


def invert_normalize_image(inpdata):
  """
  Take an image tensor, invert normalize and convert to numpy image
  """
  ts=invert_normalize(inpdata)
  a=ts.data.squeeze(0).permute(1, 2, 0).numpy()
  saveimg=(a*255.0).astype(np.uint8)
  return saveimg



def rgb2gray(rgb):
  """
  Convert RGB to grayscale image
  """
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray



def make_register_custom_cmap(cmap_colors):
  """
  Take a list of custom colors, create a matplotlib Linear Segmented Color map, register the color map and return the cmap object.
  """
  color_cmap = LinearSegmentedColormap.from_list( 'my_cmap', [ Color( c1 ).rgb for c1 in cmap_colors ] )
  plt.register_cmap(cmap = color_cmap)
  return color_cmap


def get_mpl_colormap(cmap_name):
  """
  Convert matplotlib cmap to be used for cv2 mapping
  Ref : https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image
  """
  cmap = plt.get_cmap(cmap_name)

  # Initialize the matplotlib color map
  sm = plt.cm.ScalarMappable(cmap=cmap)

  # Obtain linear color range
  color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

  return color_range.reshape(256, 1, 3)

    
def save_img_lrp_overlay_only_positive( lrp_result,  imgtensor,  q=100, title=None, outname=None):
  """
  Take image tensor and lrp result, create the heatmaps and store in `outname`
  """
  
  # Create colormaps for lrp 
  # This is a simple color map with black to orange. I use black as the pixel value is 0, 0, 0 so easy to overlay later.
  cmap_lrp = make_register_custom_cmap( [
                                        '#000000', '#ffc300', '#ffaa00',
                                          ] )

  # Normalize and smooth lrp heatmap
  hm = lrp_result.squeeze().sum(dim=0).numpy()
  hm [ hm < 0 ] = 0.0 # Only consider positive relevances
  clim = np.percentile(np.abs(hm), q)

  # Now normalize, smooth and create colormap
  final_hm = hm.copy()/clim
  final_hm = cv2.medianBlur(final_hm, 5) # Smoothen, otherwise its ugly for ResNet-50
  final_hm = cv2.applyColorMap(np.uint8(255 * final_hm), get_mpl_colormap('my_cmap'))
  final_hm = cv2.cvtColor(final_hm, cv2.COLOR_BGR2RGB)
  

  # Size of output in pixels
  h = 224
  w = 224
  my_dpi = 1200
  fig, ax = plt.subplots(1, figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)

  # Image
  img_np = invert_normalize_image(imgtensor) # No need to blur the image
  
  # Now overlay and create final visualization
  cam = cv2.addWeighted(final_hm, 0.55, img_np, 0.45, 0)
  ax.imshow(cam)
  ax.axis("off")

  # Save image as pdf
  fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
  fig.savefig("{}".format(outname), format='pdf', bbox_inches=Bbox([[0, 0], [w/my_dpi, h/my_dpi]]), dpi=my_dpi)
  plt.close()



def save_img_guided_gradcam_overlay_only_positive( gb_cam_result,  imgtensor,  q=100, title=None, outname=None):
  """
  Take image tensor and lrp result, create the heatmaps and store in `outname`
  """

  # Sum all the channels for relevance
  gb_cam_result = np.sum(gb_cam_result, axis=2, keepdims=False)

  # Create colormaps for lrp 
  # This is a simple color map with black to orange. I use black as the pixel value is 0, 0, 0 so easy to overlay later.
  cmap_lrp = make_register_custom_cmap( [
                                        '#000000', '#ffc300', '#ffaa00',
                                          ] )

  # Normalize and smooth lrp heatmap
  hm = gb_cam_result
  hm [ hm < 0 ] = 0.0 # Negative values do not make sense for gradcam
  clim = np.percentile(np.abs(hm), q)

  # Now normalize, smooth and create colormap
  final_hm = hm.copy()/clim
  #final_hm = cv2.medianBlur(final_hm, 5) # Smoothen, otherwise its ugly for ResNet-50
  final_hm = cv2.applyColorMap(np.uint8(255 * final_hm), get_mpl_colormap('my_cmap'))
  final_hm = cv2.cvtColor(final_hm, cv2.COLOR_BGR2RGB)
  

  # Size of output in pixels
  h = 224
  w = 224
  my_dpi = 1200
  fig, ax = plt.subplots(1, figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)

  # Image
  img_np = np.uint8(imgtensor*255.0) # No need to blur the image
  
  # Now overlay and create final visualization
  cam = cv2.addWeighted(final_hm, 0.55, img_np, 0.45, 0)
  ax.imshow(cam)
  ax.axis("off")

  # Save image as pdf
  fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
  fig.savefig("{}".format(outname), format='pdf', bbox_inches=Bbox([[0, 0], [w/my_dpi, h/my_dpi]]), dpi=my_dpi)
  plt.close()




def explain_save_img_lrp_overlay_only_positive( lrp_result,  imgtensor,  q=100, title=None, outname=None):
  """
  Take image tensor and lrp result, create the heatmaps and store in `outname`
  """
  
  # Create colormaps for lrp 
  # This is a simple color map with black to orange. I use black as the pixel value is 0, 0, 0 so easy to overlay later.
  cmap_lrp = make_register_custom_cmap( [
                                        '#000000', '#ffc300', '#ffaa00',
                                          ] )

  # Normalize and smooth lrp heatmap
  hm = lrp_result.squeeze().sum(dim=0).numpy()
  hm [ hm < 0 ] = 0.0 # Only consider positive relevances
  clim = np.percentile(np.abs(hm), q)

  # Now normalize, smooth and create colormap
  final_hm = hm.copy()/clim
  #final_hm = cv2.medianBlur(final_hm, 5) # Smoothen, otherwise its ugly for ResNet-50
  final_hm = cv2.applyColorMap(np.uint8(255 * final_hm), get_mpl_colormap('my_cmap'))
  final_hm = cv2.cvtColor(final_hm, cv2.COLOR_BGR2RGB)
  

  # Size of output in pixels
  h = 256
  w = 256
  my_dpi = 1200
  fig, ax = plt.subplots(1, figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)

  # Image
  img_np = invert_normalize_image(imgtensor) # No need to blur the image
  
  # Now overlay and create final visualization
  cam = cv2.addWeighted(final_hm, 0.55, img_np, 0.45, 0)
  ax.imshow(cam)
  ax.axis("off")

  # Save image as pdf
  fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0) 
  fig.savefig("{}".format(outname), format='jpg', bbox_inches=Bbox([[0, 0], [w/my_dpi, h/my_dpi]]), dpi=my_dpi)
  plt.close()

  plt.imsave("{}_image".format(outname), img_np, format='jpg')


def relevance_bounding_box(bool_mask, q=90):
  """
  Use thresholded lrp mask to obtain ROI
  """
  bool_mask = bool_mask.copy()
  bool_mask[ bool_mask<=np.percentile((bool_mask), q) ] = 0.0
  bool_mask[ bool_mask>np.percentile((bool_mask), q) ] = 1.0
  bool_mask = bool_mask.astype(np.uint8)
  itemindex= np.where(bool_mask==True)
  # print(np.min(itemindex[0]), np.min(itemindex[1]))
  # print(np.max(itemindex[0]), np.max(itemindex[1]))
  
  if len(itemindex[0]) == 0 or len(itemindex[1]) == 0:
    return bool_mask[ 0:bool_mask.shape[0], 0: bool_mask.shape[1] ], (0, bool_mask.shape[0], 0, bool_mask.shape[1])


  x_min, x_max = np.min(itemindex[0]), np.max(itemindex[0])
  y_min, y_max = np.min(itemindex[1]), np.max(itemindex[1])

  return bool_mask[ x_min:x_max, y_min:y_max ], (x_min, x_max, y_min, y_max)



def extract_patch( lrp_result,  imgtensor,  q=100, title=None, outname=None):
  """
  Take image tensor and lrp result, create the heatmaps and store in `outname`
  """
  # Create colormaps for lrp 
  # This is a simple color map with black to orange. I use black as the pixel value is 0, 0, 0 so easy to overlay later.
  cmap_lrp = make_register_custom_cmap( [
                                        '#000000', '#ffc300', '#ffaa00',
                                          ] )

  # Normalize and smooth lrp heatmap
  hm = lrp_result.squeeze().sum(dim=0).numpy()
  hm [ hm < 0 ] = 0.0 # Only consider positive relevances
  clim = np.percentile(np.abs(hm), q)

  # Now normalize, smooth and create colormap
  final_hm = hm.copy()/clim

  # Extract patch
  _, (x_min, x_max, y_min, y_max) = relevance_bounding_box(final_hm, q=75)
  # print( (x_min, x_max, y_min, y_max) )

  # Image
  img_np = invert_normalize_image(imgtensor) [ x_min:x_max, y_min:y_max ] # No need to blur the image

  # Convert to PIL
  pil_image = Image.fromarray(img_np)
  pil_image.save("{}".format(outname), quality=95)