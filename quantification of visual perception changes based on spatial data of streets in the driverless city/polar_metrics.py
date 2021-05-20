# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:01:09 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import pickle
from database import postSQL2gpd,gpd2postSQL
from segs_object_analysis import idx_clustering

def polar_metrics(img_root,coords,hsv_lower,hsv_upper):
    from tqdm import tqdm
    import glob,os 
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import rasterio as rio
    from rasterio.transform import from_origin
    from pathlib import Path 
    import pylandstats as pls
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj    
    
    
    polar_seg_fns=glob.glob(os.path.join(img_root,'*.jpg'))
    # print(polar_seg_fns)
    hsv_lower_=np.asarray(hsv_lower)
    hsv_upper_=np.asarray(hsv_upper)
    
    transform=from_origin(472137, 5015782, 100, 100)  #472137, 5015782, 0.5, 0.5
    
    '''
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
    '''
    metrics=['total_area','area_mn','perimeter_mn','perimeter_area_ratio_mn','number_of_patches','landscape_shape_index','shape_index_mn','fractal_dimension_mn',]
    columns=["fn_stem","fn_key","fn_idx","geometry",]+metrics
    
    sky_class_level_metrics=pd.DataFrame(columns=columns)    
    i=0
    for fn in tqdm(polar_seg_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")    
        coord=coords[fn_key][int(fn_idx)]
        # print(fn)
        img=cv2.imread(fn)
        # print(img)
        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # print(np.unique(img_hsv))
        mask=cv2.inRange(img_hsv, hsv_lower_, hsv_upper_)
        # print(mask.shape)        
        # plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
        # plt.show()
        
        mask=np.where(mask==255,1,mask)#.astype(np.float64)
        # mask=np.where(mask==0,0.0,mask)
        # print(mask)
        # print(np.unique(mask))
        # print(mask.dtype)        
        tiff_fn=os.path.join('./processed data/tiff_sky','{}.tif'.format(Path(fn).stem))
        # tiff_fn=os.path.join('./processed data/tiff_sky','temp.tif')
        dst=rio.open(tiff_fn, 'w', driver='GTiff',
                                  height=mask.shape[0], width=mask.shape[1],
                                  count=1, dtype=str(mask.dtype),#dtype=rio.uint8,
                                  crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                                  transform=transform)        
        dst.nodata=0
        dst.write(mask,1)
        dst.close()
        # print(mask.dtype)
        
        ls=pls.Landscape(tiff_fn)
        # class_metrics_df=ls.compute_class_metrics_df()        
        # # print(class_metrics_df)
        # class_metrics_dict=class_metrics_df.transpose().to_dict()[1]
        try:
            # class_metrics_dict={
            # 'total_area':ls.total_area() ,
            # 'number_of_patches':ls.number_of_patches(),
            # 'landscape_shape_index':ls.landscape_shape_index(),
            # 'shape_index_mn':ls.shape_index_mn(),
            # 'fractal_dimension_mn':ls.fractal_dimension_mn(),
            # }        
            class_metrics_df=ls.compute_class_metrics_df(metrics=metrics) 
            class_metrics_dict=class_metrics_df.transpose().to_dict()[1]
        except:
            class_metrics_dict={k:0 for k in metrics}              
        
        class_metrics_dict.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})
        sky_class_level_metrics=sky_class_level_metrics.append(class_metrics_dict,ignore_index=True)
        
        # if i==3:break
        # i+=1

    wgs84='EPSG:4326' #pyproj.CRS('EPSG:4326')
    sky_class_level_metrics_gdf=gpd.GeoDataFrame(sky_class_level_metrics,geometry=sky_class_level_metrics.geometry,crs=wgs84) 
    return sky_class_level_metrics_gdf

# deprecated
# packages\pylandstats\landscape.py has not been compiled for Transonic-Numba

'''
with open('./processed data/coords.pkl','rb') as f:
    coords=pickle.load(f) 
hsv_lower=[0,0,200]
hsv_upper=[180,255,255]    
def polar_metrics_single(fn):
    from tqdm import tqdm
    import glob,os 
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import rasterio as rio
    from rasterio.transform import from_origin
    from pathlib import Path 
    import pylandstats as pls
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj    
    
    hsv_lower_=np.asarray(hsv_lower)
    hsv_upper_=np.asarray(hsv_upper)
    
    transform=from_origin(472137, 5015782, 100, 100)  #472137, 5015782, 0.5, 0.5
    


    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")    
    coord=coords[fn_key][int(fn_idx)]
    # print(fn)
    img=cv2.imread(fn)
    # print(img)
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(np.unique(img_hsv))
    mask=cv2.inRange(img_hsv, hsv_lower_, hsv_upper_)
    # print(mask.shape)        
    # plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
    # plt.show()
    
    mask=np.where(mask==255,1,mask)#.astype(np.float64)
    # mask=np.where(mask==0,0.0,mask)
    # print(mask)
    # print(np.unique(mask))
    # print(mask.dtype)        
    tiff_fn=os.path.join('./processed data/tiff_sky','{}.tif'.format(Path(fn).stem))
    # tiff_fn=os.path.join('./processed data/tiff_sky','temp.tif')
    dst=rio.open(tiff_fn, 'w', driver='GTiff',
                              height=mask.shape[0], width=mask.shape[1],
                              count=1, dtype=str(mask.dtype),#dtype=rio.uint8,
                              crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                              transform=transform)        
    dst.nodata=0
    dst.write(mask,1)
    dst.close()
    # print(mask.dtype)
    
    ls=pls.Landscape(tiff_fn)
    class_metrics_df=ls.compute_class_metrics_df()
    # print(class_metrics_df)
    class_metrics_dict=class_metrics_df.transpose().to_dict()[1]
    class_metrics_dict.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})
    
    return class_metrics_dict

    

def polar_metrics(img_root):
    from tqdm import tqdm
    import glob,os 
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import rasterio as rio
    from rasterio.transform import from_origin
    from pathlib import Path 
    
    polar_seg_fns=glob.glob(os.path.join(img_root,'*.jpg'))
    
    transform=from_origin(472137, 5015782, 1, 1)  #472137, 5015782, 0.5, 0.5
    for fn in tqdm(polar_seg_fns):
        img=cv2.imread(fn)
        # print(img)
        mask=np.all(img==(0,0,0),axis=-1)*1
        # plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
        # plt.show()        
        # print(mask)
        tiff_fn=os.path.join('./processed data/tiff_sky','{}.tif'.format(Path(fn).stem))
        dst=rio.open(tiff_fn, 'w', driver='GTiff',
                                  height=mask.shape[0], width=mask.shape[1],
                                  count=1, dtype=str(mask.dtype),#dtype=rio.uint8,
                                  crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                                  transform=transform)        
        dst.nodata=0
        dst.write(mask,1)
        dst.close()
        print(mask.dtype)
        break        
'''

def correlation_df(df,idxes):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    corr=df[idxes].corr()
    print(corr)
    corr.to_excel('./graph/sky index corr.xlsx')
    
    # Generate a mask for the upper triangle
    mask=np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax=plt.subplots(figsize=(11, 9))    
    
    # Generate a custom diverging colormap
    cmap=sns.diverging_palette(230, 20, as_cmap=True)    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})    
        

if __name__=="__main__":
    with open('./processed data/coords_tourLine.pkl','rb') as f: #'./processed data/coords.pkl'
        coords=pickle.load(f) 
    
    polar_seg_root='./processed data/tourline_polar_sky' #'./processed data/polar_sky'
    # In OpenCV, for HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
    sky_class_level_metrics_gdf=polar_metrics(polar_seg_root,coords,hsv_lower=[0,0,200],hsv_upper=[180,255,255],) #hsv_lower=[90,100,0],hsv_upper=[110,200,255] /hsv_lower=[0,0,0],hsv_upper=[50,50,50]
    # gpd2postSQL(sky_class_level_metrics_gdf,table_name='sky_class_level_metrics',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    gpd2postSQL(sky_class_level_metrics_gdf,table_name='tl_sky_class_level_metrics',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
   
    '''
    #for black
    cvInRangeS(imgHSV, cvScalar(0, 0, 0, 0), cvScalar(180, 255, 30, 0), imgThreshold);    
    #for white
    cvInRangeS(imgHSV, cvScalar(0, 0, 200, 0), cvScalar(180, 255, 255, 0), imgThreshold);       
    
    '''
    
    # sky_class_level_metrics_gdf=postSQL2gpd(table_name='sky_class_level_metrics',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # sky_class_level_metrics_gdf=idx_clustering(sky_class_level_metrics_gdf,field='fractal_dimension_mn',n_clusters=32)
    # gpd2postSQL(sky_class_level_metrics_gdf,table_name='sky_class_level_metrics',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    # sky_class_level_metrics_gdf=postSQL2gpd(table_name='sky_class_level_metrics',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # idxes=['number_of_patches','perimeter_area_ratio_mn','shape_index_mn', 'fractal_dimension_mn',]    
    # correlation_df(sky_class_level_metrics_gdf,idxes)


