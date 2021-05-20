# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:14:17 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import glob,os
from multiprocessing import Pool
from tqdm import tqdm
from equi_to_cube_pool import labels_equi2cube_pool


if __name__ == '__main__':    
    label_seg_path=r'./processed data/tourline_label_seg'   #r'./processed data/label_seg'
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    with Pool(8) as p:
        p.map(labels_equi2cube_pool, tqdm(label_seg_fns))    