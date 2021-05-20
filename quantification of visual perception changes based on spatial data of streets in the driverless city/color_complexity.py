# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:23:09 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from color_complexity_pool import find_dominant_colors_pool,dominant2cluster_colors_pool
from multiprocessing import Pool
from tqdm import tqdm
import glob,os 
import pickle
import pandas as pd
from shapely.geometry import Point
import pyproj
import geopandas as gpd
from database import postSQL2gpd,gpd2postSQL

def find_dominant_colors_pool_main(img_path):
    img_fns=glob.glob(os.path.join(img_path,'*.jpg'))
    with Pool(8) as p:
        color_dic_list=p.map(find_dominant_colors_pool, tqdm(img_fns))
    
    number_of_colors=16
    img_dominant_color=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    for color_dic in color_dic_list:    
        img_dominant_color=img_dominant_color.append(color_dic,ignore_index=True)
    
    wgs84=pyproj.CRS('EPSG:4326')
    img_dominant_color_gdf=gpd.GeoDataFrame(img_dominant_color,geometry=img_dominant_color.geometry,crs=wgs84)
    #'img_dominant_color'
    gpd2postSQL(img_dominant_color_gdf,table_name='tl_img_dominant_color',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    
def hex2rgb(hex_str):
    from PIL import ImageColor
    return ImageColor.getcolor(hex_str, "RGB")   

def idx_clustering_dims(gdf,field,n_clusters=10):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    from sklearn import preprocessing    
    
    pts_geometry=gdf[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # print(pts_geometry)
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    # print(pts_coordis)
    
    nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
    connectivity=nbrs.kneighbors_graph(pts_coordis)
    # print(connectivity.toarray())
    
    le=preprocessing.LabelEncoder()
    # X=np.expand_dims(panorama_object_percent_gdf[field].to_numpy(),axis=1)
    X_=gdf[field].to_numpy()
    # print(X_)
    hex2rgb_func=np.vectorize(hex2rgb)
    X=np.stack(hex2rgb_func(X_.flatten())).T
    X=X.reshape((X_.shape[0],X_.shape[1]*3))
    # X_flatten=X_.flatten()
    # le.fit(X_flatten)
    # X=le.transform(X_flatten).reshape(X_.shape)
       
    # print(X.shape)
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    print(clustering.labels_.shape)
    gdf['clustering']=clustering.labels_
    
    return gdf    

def dominant2cluster_colors_pool_main(img_path):
    img_fns=glob.glob(os.path.join(img_path,'*.jpg'))
    # print(img_fns)
    with Pool(8) as p:
        color_dic_list=p.map(dominant2cluster_colors_pool, tqdm(img_fns))
    # print(color_dic_list)
    with open('./processed data/tl_colors_dominant2cluster.pkl','wb') as f: #'./processed data/colors_dominant2cluster.pkl'
        pickle.dump(color_dic_list,f) 
        
def colors_entropy(colors_dominant_clustering_fn):
    from tqdm import tqdm
    import math,copy
    
    with open(colors_dominant_clustering_fn,'rb') as f:
        colors_dominant_clustering=pickle.load(f)#[:10]
    # print(sum(colors_dominant_clustering[0]['counter'].values()))
    def entropy(counter_dict):
        percentage=[i/sum(counter_dict.values()) for i in counter_dict.values()]
        ve=0.0
        for perc in percentage:
            if perc!=0.:
                ve-=perc*math.log(perc)            
        max_entropy=math.log(len(counter_dict.keys()))
        frank_e=ve/max_entropy*100    
        return frank_e
            
    color_dominant_entropy=[entropy(dic['counter']) for dic in tqdm(colors_dominant_clustering)]
    # print(color_dominant_entropy)
    wgs84=pyproj.CRS('EPSG:4326')
    colors_dominant_clustering_copy=copy.deepcopy(colors_dominant_clustering)
    [colors_dominant_clustering_copy[i].update({'counter':color_dominant_entropy[i]}) for i in range(len(colors_dominant_clustering_copy))]
    
    colors_dominant_entropy=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",'counter'])
    for colors_dominant_dic in colors_dominant_clustering_copy:    
        colors_dominant_entropy=colors_dominant_entropy.append(colors_dominant_dic,ignore_index=True)        
    
    colors_dominant_entropy_gdf=gpd.GeoDataFrame(colors_dominant_entropy,geometry=colors_dominant_entropy.geometry,crs=wgs84)
    gpd2postSQL(colors_dominant_entropy_gdf,table_name='tl_colors_dominant_entropy',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
        
    # return colors_dominant_clustering_copy

if __name__=="__main__":
    img_path=r'./data/panoramic imgs_tour line valid'  #r'./data/panoramic imgs valid'
    # find_dominant_colors_pool_main(img_path) #tl_img_dominant_color
    #'img_dominant_color'
    # img_dominant_color_gdf=postSQL2gpd(table_name='tl_img_dominant_color',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # clustering_dominant_colors_gdf=idx_clustering_dims(img_dominant_color_gdf,field=[ '0', '1', '2', '3', '4', '5','6', '7'],n_clusters=10)
    #'clustering_dominant_color'
    # gpd2postSQL(clustering_dominant_colors_gdf,table_name='tl_clustering_dominant_color',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    # img_dominant_color_gdf=postSQL2gpd(table_name='img_dominant_color',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    # colors entropy    
    # dominant2cluster_colors_pool_main(img_path)
    
    colors_entropy('./processed data/tl_colors_dominant2cluster.pkl')
    
    colors_dominant_entropy_gdf=postSQL2gpd(table_name='tl_colors_dominant_entropy',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    #frequency table by range    
    # bins=[0,50,65,75,100]    
    # frequency=colors_dominant_entropy_gdf[['counter']].apply(pd.Series.value_counts,bins=bins,)
    # frequency['percentage']=(frequency['counter'] / frequency['counter'].sum()) * 100
