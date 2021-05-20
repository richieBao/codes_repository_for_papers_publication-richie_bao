# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:11:12 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from multiprocessing import Pool
from panorama_equirectangular2polar_pool import panorama_equirectangular2polar_single
from tqdm import tqdm
import glob,os 

if __name__=="__main__":
    # imgs_root='./processed data/img_seg'   #'./data/sample/v2.jfif'
    imgs_root='./data/panoramic imgs valid'
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))
    with Pool(8) as p:
        p.map(panorama_equirectangular2polar_single, tqdm(img_fns))