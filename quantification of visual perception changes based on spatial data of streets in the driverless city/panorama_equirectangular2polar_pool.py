# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:38:23 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
ref:http://www.richwareham.com/little-planet-projection/
"""
#Stereographic Projection  /Stereographic 'little/tiny planet' 
import numpy as np

def output_coord_to_r_theta(coords):
    """Convert co-ordinates in the output image to r, theta co-ordinates.
    The r co-ordinate is scaled to range from from 0 to 1. The theta
    co-ordinate is scaled to range from 0 to 1.
    
    A Nx2 array is returned with r being the first column and theta being
    the second.
    """
    # Calculate x- and y-co-ordinate offsets from the centre:
    x_offset = coords[:,0] - (output_shape[1]/2)
    y_offset = coords[:,1] - (output_shape[0]/2)
    
    # Calculate r and theta in pixels and radians:
    r = np.sqrt(x_offset ** 2 + y_offset ** 2)
    theta = np.arctan2(y_offset, x_offset)
    
    # The maximum value r can take is the diagonal corner:
    max_x_offset, max_y_offset = output_shape[1]/2, output_shape[0]/2
    max_r = np.sqrt(max_x_offset ** 2 + max_y_offset ** 2)
    
    # Scale r to lie between 0 and 1
    r = r / max_r
    
    # arctan2 returns an angle in radians between -pi and +pi. Re-scale
    # it to lie between 0 and 1
    theta = (theta + np.pi) / (2*np.pi)
    
    # Stack r and theta together into one array. Note that r and theta are initially
    # 1-d or "1xN" arrays and so we vertically stack them and then transpose
    # to get the desired output.
    return np.vstack((r, theta)).T

def r_theta_to_input_coords(r_theta):
    """Convert a Nx2 array of r, theta co-ordinates into the corresponding
    co-ordinates in the input image.
    
    Return a Nx2 array of input image co-ordinates.
    
    """
    # Extract r and theta from input
    r, theta = r_theta[:,0], r_theta[:,1]
    
    # Theta wraps at the side of the image. That is to say that theta=1.1
    # is equivalent to theta=0.1 => just extract the fractional part of
    # theta
    theta = theta - np.floor(theta)
    
    # Calculate the maximum x- and y-co-ordinates
    max_x, max_y = input_shape[1]-1, input_shape[0]-1
    
    # Calculate x co-ordinates from theta
    xs = theta * max_x
    
    # Calculate y co-ordinates from r noting that r=0 means maximum y
    # and r=1 means minimum y
    ys = (1-r) * max_y
    
    # Return the x- and y-co-ordinates stacked into a single Nx2 array
    return np.hstack((xs, ys))

def little_planet_1(coords):
    """Chain our two mapping functions together."""
    r_theta = output_coord_to_r_theta(coords)
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def little_planet_2(coords):
    """Chain our two mapping functions together with modified r."""
    r_theta = output_coord_to_r_theta(coords)
    # Take square root of r
    r_theta[:,0] = np.sqrt(r_theta[:,0])
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def little_planet_3(coords):
    """Chain our two mapping functions together with modified r
    and shifted theta.
    
    """
    r_theta = output_coord_to_r_theta(coords)
    
    # Take square root of r
    r_theta[:,0] = np.sqrt(r_theta[:,0])
    
    # Shift theta
    r_theta[:,1] += 0.1
    
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def little_planet_4(coords):
    """Chain our two mapping functions together with modified and
    scaled r and shifted theta.
    
    """
    r_theta = output_coord_to_r_theta(coords)
    
    # Scale r down a little to zoom in
    r_theta[:,0] *= 0.75
    
    # Take square root of r
    r_theta[:,0] = np.sqrt(r_theta[:,0])
    
    # Shift theta
    r_theta[:,1] += 0.1
    
    input_coords = r_theta_to_input_coords(r_theta)
    return input_coords

def panorama_equirectangular2polar(imgs_root,little_planet,output_shape):
    import glob,os 
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps
    import numpy as np
    from skimage.transform import warp
    import matplotlib
    from PIL import Image
    from pathlib import Path
    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))

    # print(img_fns)
    for fn in tqdm(img_fns):        
        pano=np.asarray(ImageOps.flip(Image.open(fn)))
        global input_shape
        input_shape = pano.shape
        # print(input_shape)        
        pano_warp=warp(pano, little_planet, output_shape=output_shape)
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(pano_warp)
        
        # print(pano_warp)
        # The image is a NxMx3 array of floating point values from 0 to 1. Convert this to
        # bytes from 0 to 255 for saving the image:
        pano_warp = (255 * pano_warp).astype(np.uint8)       
        # print(pano_warp)
        im=Image.fromarray(pano_warp)
        # im_save_fn=os.path.join('./processed data/polar_img','{}.jpg'.format(Path(fn).stem))
        im_save_fn=os.path.join('./processed data/polar_seg','{}.jpg'.format(Path(fn).stem))
        im.save(im_save_fn)
        # print(im_save_fn)
        # mask = (b == [70,130,179]).all(-1)
        # ii=Image.fromarray(mask.astype('uint8')*255)
        
        # break

    # return im,pano_warp
    
output_shape = (1024,1024)   
little_planet= little_planet_1
def panorama_equirectangular2polar_single(fn,): #little_planet,output_shape
    import glob,os 
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps
    import numpy as np
    from skimage.transform import warp
    import matplotlib
    from PIL import Image
    from pathlib import Path   
       
    pano=np.asarray(ImageOps.flip(Image.open(fn)))
    global input_shape
    input_shape = pano.shape     
    pano_warp=warp(pano, little_planet, output_shape=output_shape)
    pano_warp = (255 * pano_warp).astype(np.uint8)    
    im=Image.fromarray(pano_warp)
    im_save_fn=os.path.join('./processed data/polar_img','{}.jpg'.format(Path(fn).stem))
    im.save(im_save_fn)    
    
def sky_equirectangular2polar_single(label_seg_fn):
    import glob,os 
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps
    import numpy as np
    from skimage.transform import warp
    import matplotlib
    from PIL import Image
    from pathlib import Path   
    import pickle
    from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
       
    # pano=np.asarray(ImageOps.flip(Image.open(fn)))
    with open(label_seg_fn,'rb') as f:
        label_seg=pickle.load(f).numpy() 
    # print(label_seg)  
    # pano=np.where(label_seg==22,1,0)     
    sky_bool=label_seg==22
    sky_img=ndarray_to_pil(sky_bool).convert("1")
    pano=np.asarray(ImageOps.flip(sky_img))
    
    global input_shape
    input_shape = pano.shape     
    pano_warp=warp(pano, little_planet, output_shape=output_shape)
    pano_warp = (255 * pano_warp).astype(np.uint8)    
    im=Image.fromarray(pano_warp)
    im_save_fn=os.path.join('./processed data/tourline_polar_sky','{}.jpg'.format(Path(label_seg_fn).stem)) #'./processed data/polar_sky'
    im.save(im_save_fn)     

if __name__=="__main__":
    imgs_root='./processed data/img_seg_redefined_color'   #'./data/sample/v2.jfif'
    # imgs_root='./data/panoramic imgs valid'
    output_shape = (1024,1024)
    panorama_equirectangular2polar(imgs_root,little_planet_1,output_shape)