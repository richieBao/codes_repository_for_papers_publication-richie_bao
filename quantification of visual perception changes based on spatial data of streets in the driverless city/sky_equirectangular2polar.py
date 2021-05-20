# -*- coding: utf-8 -*-
"""
Created on Sun May  2 20:20:09 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from panorama_equirectangular2polar_pool import sky_equirectangular2polar_single
import glob,os 
from multiprocessing import Pool
from tqdm import tqdm


def sky_polar(label_seg_path):
    from tqdm import tqdm
    import glob,os 
    import pickle
    import numpy as np
    from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
    from PIL import Image,ImageOps
    
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    # print(label_seg_fns)
    for label_seg_fn in tqdm(label_seg_fns):
        with open(label_seg_fn,'rb') as f:
            label_seg=pickle.load(f).numpy() 
        # print(label_seg)  
        # sky=np.where(label_seg==22,1,0)
        sky_bool=label_seg==22
        # print(sky)
        sky_img=ndarray_to_pil(sky_bool).convert("1")
        # print(sky_img)
        pano=np.asarray(ImageOps.flip(sky_img))
        print(pano)
        # sky_img
        
        break
    
    return sky_img





if __name__=="__main__":
    label_seg_path=r'./processed data/tourline_label_seg' #r'./processed data/label_seg'
    # a=sky_polar(label_seg_path)
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    with Pool(8) as p:
        p.map(sky_equirectangular2polar_single, tqdm(label_seg_fns))    