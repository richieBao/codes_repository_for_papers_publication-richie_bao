# -*- coding: utf-8 -*-
"""
Created on Sun May  2 22:57:59 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from multiprocessing import Pool
from polar_metrics_pool import polar_metrics_single
from tqdm import tqdm
import glob,os 
import pandas as pd
#packages\pylandstats\landscape.py has not been compiled for Transonic-Numba

columns=["fn_stem","fn_key","fn_idx","geometry",]+['total_area', 'proportion_of_landscape', 'number_of_patches',
   'patch_density', 'largest_patch_index', 'total_edge', 'edge_density',
   'landscape_shape_index', 'effective_mesh_size', 'area_mn', 'area_am',
   'area_md', 'area_ra', 'area_sd', 'area_cv', 'perimeter_mn',
   'perimeter_am', 'perimeter_md', 'perimeter_ra', 'perimeter_sd',
   'perimeter_cv', 'perimeter_area_ratio_mn', 'perimeter_area_ratio_am',
   'perimeter_area_ratio_md', 'perimeter_area_ratio_ra',
   'perimeter_area_ratio_sd', 'perimeter_area_ratio_cv', 'shape_index_mn',
   'shape_index_am', 'shape_index_md', 'shape_index_ra', 'shape_index_sd',
   'shape_index_cv', 'fractal_dimension_mn', 'fractal_dimension_am',
   'fractal_dimension_md', 'fractal_dimension_ra', 'fractal_dimension_sd',
   'fractal_dimension_cv', 'euclidean_nearest_neighbor_mn',
   'euclidean_nearest_neighbor_am', 'euclidean_nearest_neighbor_md',
   'euclidean_nearest_neighbor_ra', 'euclidean_nearest_neighbor_sd',
   'euclidean_nearest_neighbor_cv']
sky_class_level_metrics=pd.DataFrame(columns=columns)   

if __name__=="__main__":
    # imgs_root='./processed data/img_seg'   #'./data/sample/v2.jfif'
    polar_seg_root='./processed data/polar_sky'
    polar_seg_fns=glob.glob(os.path.join(polar_seg_root,'*.jpg'))
    with Pool(8) as p:
        class_metrics=p.map(polar_metrics_single, tqdm(polar_seg_fns[:2]))  


    
    # sky_class_level_metrics=sky_class_level_metrics.append(class_metrics_dict,ignore_index=True)
    
    # wgs84='EPSG:4326' #pyproj.CRS('EPSG:4326')
    # sky_class_level_metrics_gdf=gpd.GeoDataFrame(sky_class_level_metrics,geometry=sky_class_level_metrics.geometry,crs=wgs84) 
  