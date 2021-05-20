# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:21:17 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import pickle
from database import postSQL2gpd,gpd2postSQL
import pandas as pd

xian_epsg=32649 #Xi'an   WGS84 / UTM zone 49N
wgs84_epsg=4326

'''
#IoU of 27 classes
print ("pole    : %.6f" % (iou_classes[0]*100.0), "%\t")
print ("slight  : %.6f" % (iou_classes[1]*100.0), "%\t")
print ("bboard  : %.6f" % (iou_classes[2]*100.0), "%\t")
print ("tlight  : %.6f" % (iou_classes[3]*100.0), "%\t")
print ("car     : %.6f" % (iou_classes[4]*100.0), "%\t")
print ("truck   : %.6f" % (iou_classes[5]*100.0), "%\t")
print ("bicycle : %.6f" % (iou_classes[6]*100.0), "%\t")
print ("motor   : %.6f" % (iou_classes[7]*100.0), "%\t")
print ("bus     : %.6f" % (iou_classes[8]*100.0), "%\t")
print ("tsignf  : %.6f" % (iou_classes[9]*100.0), "%\t")
print ("tsignb  : %.6f" % (iou_classes[10]*100.0), "%\t")
print ("road    : %.6f" % (iou_classes[11]*100.0), "%\t")
print ("sidewalk: %.6f" % (iou_classes[12]*100.0), "%\t")
print ("curbcut : %.6f" % (iou_classes[13]*100.0), "%\t")
print ("crosspln: %.6f" % (iou_classes[14]*100.0), "%\t")
print ("bikelane: %.6f" % (iou_classes[15]*100.0), "%\t")
print ("curb    : %.6f" % (iou_classes[16]*100.0), "%\t")
print ("fence   : %.6f" % (iou_classes[17]*100.0), "%\t")
print ("wall    : %.6f" % (iou_classes[18]*100.0), "%\t")
print ("building: %.6f" % (iou_classes[19]*100.0), "%\t")
print ("person  : %.6f" % (iou_classes[20]*100.0), "%\t")
print ("rider   : %.6f" % (iou_classes[21]*100.0), "%\t")
print ("sky     : %.6f" % (iou_classes[22]*100.0), "%\t")
print ("vege    : %.6f" % (iou_classes[23]*100.0), "%\t")
print ("terrain : %.6f" % (iou_classes[24]*100.0), "%\t")
print ("markings: %.6f" % (iou_classes[25]*100.0), "%\t")
print ("crosszeb: %.6f" % (iou_classes[26]*100.0), "%\t")

def colormap_mapillary(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([153,153,153])
    cmap[1,:] = np.array([210,170,100])
    cmap[2,:] = np.array([220,220,220])
    cmap[3,:] = np.array([250,170, 30])
    cmap[4,:] = np.array([  0,  0,142]) #car
    cmap[5,:] = np.array([  0,  0, 70])

    cmap[6,:] = np.array([119, 11, 32])
    cmap[7,:] = np.array([  0,  0,230])
    cmap[8,:] = np.array([  0, 60,100])
    cmap[9,:] = np.array([220,220,  0])
    cmap[10,:]= np.array([192,192,192])

    cmap[11,:]= np.array([128, 64,128]) #road
    cmap[12,:]= np.array([244, 35,232]) #sidewalk
    cmap[13,:]= np.array([170,170,170])
    cmap[14,:]= np.array([140,140,200])
    cmap[15,:]= np.array([128, 64,255])

    cmap[16,:]= np.array([196,196,196]) #curb
    cmap[17,:]= np.array([190,153,153])
    cmap[18,:]= np.array([102,102,156])
    cmap[19,:]= np.array([ 70, 70, 70])

    cmap[20,:]= np.array([220, 20, 60]) #person
    cmap[21,:]= np.array([255,  0,  0])
    cmap[22,:]= np.array([ 70,130,180]) #sky
    cmap[23,:]= np.array([107,142, 35]) #green/tree
 
    cmap[24,:]= np.array([152,251,152])
    cmap[25,:]= np.array([255,255,255])
    cmap[26,:]= np.array([200,128,128]) #crosswalk
    cmap[27,:]= np.array([  0,  0,  0]) #Nan
    
    return cmap
'''

def seg_equirectangular_idxs(label_seg_path,img_Seg_path,img_path,coords):
    import glob,os    
    import pickle
    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj
    
    panorama_object_num=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(28)))
    label_mapping={
        0:"pole",
        1:"slight",
        2:"bboard",
        3:"tlight",
        4:"car",
        5:"truck",
        6:"bicycle",
        7:"motor",
        8:"bus",
        9:"tsignf",
        10:"tsignb",
        11:"road",
        12:"sidewalk",
        13:"curbcut",
        14:"crosspln",
        15:"bikelane",
        16:"curb",
        17:"fence",
        18:"wall",
        19:"building",
        20:"person",
        21:"rider",
        22:"sky",
        23:"vege",
        24:"terrain",
        25:"markings",
        26:"crosszeb",
        27:"Nan",                           
        }
    
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    # print(label_seg_fns)
    i=0
    for label_seg_fn in tqdm(label_seg_fns):
        with open(label_seg_fn,'rb') as f:
            label_seg=pickle.load(f)  
        # print("\n img shape={}".format(label_seg.shape))
        # print(label_seg.numpy())  
        # print(label_seg_fn)
        fn_stem=Path(label_seg_fn).stem
        fn_key,fn_idx=fn_stem.split("_")
        # print(fn_stem,fn_key,fn_idx)
        # with Image.open(os.path.join(img_Seg_path,fn_stem+'.jpg')) as im_seg:
        #     im_seg.show()
        # with Image.open(os.path.join(img_path,fn_stem+'.jpg')) as im:
        #     im.show()        
        
        unique_elements, counts_elements=np.unique(label_seg, return_counts=True)
        object_frequency=dict(zip(unique_elements, counts_elements))
        # print(object_frequency)
        object_frequency_update={k:object_frequency[k] if k in object_frequency.keys() else 0 for k in range(28) }
        coord=coords[fn_key][int(fn_idx)]
        object_frequency_update.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})
        # print(object_frequency_update)
        panorama_object_num=panorama_object_num.append(object_frequency_update,ignore_index=True)
        # print(panorama_object_num)   
        
        # if i==0:break
        # i+=1
    # print(np.prod(label_seg.shape))
    panorama_object_percent=panorama_object_num.copy(deep=True)
    panorama_object_percent[list(range(28))]=panorama_object_num[list(range(28))].div(np.prod(label_seg.shape)/100)    
    
    # print(panorama_object_percent.sum(axis=1))
    # print(panorama_object_percent)

    # panorama_object_num=panorama_object_num.rename(columns=label_mapping)
    # print(panorama_object_num)
    panorama_object_percent=panorama_object_percent.rename(columns=label_mapping)
    # print(panorama_object_percent)
    wgs84=pyproj.CRS('EPSG:4326')
    panorama_object_percent_gdf=gpd.GeoDataFrame(panorama_object_percent,geometry=panorama_object_percent.geometry,crs=wgs84) 
    
    print(panorama_object_percent_gdf)
    return panorama_object_percent_gdf


def visual_entropy(panorama_object_percent_gdf):
    panorama_object_percent_gdf['ground']=panorama_object_percent_gdf.apply(lambda row:100-row.sky-row.vege-row.building,axis=1)
    def ve_row(row):
        import math
        import pandas as pd
        label=['pole', 'slight', 'bboard', 'tlight', 'car', 'truck', 'bicycle', 'motor', 'bus', 'tsignf', 'tsignb', 'road', 'sidewalk', 'curbcut', 'crosspln', 'bikelane', 'curb', 'fence', 'wall', 'building', 'person', 'rider', 'sky', 'vege', 'terrain', 'markings', 'crosszeb', 'Nan']
        ve=0.0
        for i in label:
            decimal_percentage=row[i]/100
            # print(decimal_percentage)
            if decimal_percentage!=0.:
                ve-=decimal_percentage*math.log(decimal_percentage)
        max_entropy=math.log(len(label))
        frank_e=ve/max_entropy*100
        
        return pd.Series([ve,frank_e])
    
    panorama_object_percent_gdf[['ve','equilibrium_degree']]=panorama_object_percent_gdf.apply(ve_row,axis=1)
    # print(panorama_object_percent_gdf)    
    return panorama_object_percent_gdf

def idx_clustering(panorama_object_percent_gdf,field,n_clusters=10):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    
    pts_geometry=panorama_object_percent_gdf[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # print(pts_geometry)
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    # print(pts_coordis)
    
    nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
    connectivity=nbrs.kneighbors_graph(pts_coordis)
    # print(connectivity.toarray())
    
    X=np.expand_dims(panorama_object_percent_gdf[field].to_numpy(),axis=1)
    # print(X)
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    # print(clustering.labels_.shape)
    panorama_object_percent_gdf['clustering_'+field]=clustering.labels_
    
    mean=panorama_object_percent_gdf.groupby(['clustering_'+field])[field].mean() #.reset_index()
    panorama_object_percent_gdf['clustering_'+field+'_mean']=panorama_object_percent_gdf['clustering_'+field].map(mean.to_dict())
    
    return panorama_object_percent_gdf
    
    

if __name__=="__main__":
    # with open('./processed data/pt_fns.pkl','rb') as f:
    #     img_fns=pickle.load(f)    
    # with open('./processed data/imgVal_fns.pkl','rb') as f:
    #     imgVal_fns=pickle.load(f) 
    with open('./processed data/coords_tourLine.pkl','rb') as f: #'./processed data/coords.pkl'
        coords=pickle.load(f)        
        
    # pts_gdf=postSQL2gpd(table_name='pts',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    #A-equirectangular
    label_seg_path=r'./processed data/tourline_label_seg'    #r'./processed data/label_seg'     
    img_Seg_path=r'./processed data/tourline_img_seg_redefined_color' #r'./processed data/img_seg'
    img_path=r'./data/panoramic imgs_tour line valid' #r'./data/panoramic imgs'
    panorama_object_percent_gdf=seg_equirectangular_idxs(label_seg_path,img_Seg_path,img_path,coords,)
    panorama_object_percent_gdf.plot()
    # gpd2postSQL(panorama_object_percent_gdf,table_name='panorama_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    gpd2postSQL(panorama_object_percent_gdf,table_name='tourLine_panorama_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    
    # panorama_object_percent_gdf=postSQL2gpd(table_name='panorama_object_percent',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    panorama_object_percent_gdf=visual_entropy(panorama_object_percent_gdf)   
    # gpd2postSQL(panorama_object_percent_gdf,table_name='panorama_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # panorama_object_percent_gdf=idx_clustering(panorama_object_percent_gdf,field='equilibrium_degree',n_clusters=10)
    # gpd2postSQL(panorama_object_percent_gdf,table_name='panorama_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
        
    
    #B-cube
    # label_seg_path=r'./processed data/label_seg_cube'        
    # img_Seg_path=r'./processed data/img_seg_cube'
    # img_path=r'./data/panoramic imgs valid'
    
    # cube_object_percent_gdf=seg_equirectangular_idxs(label_seg_path,img_Seg_path,img_path,coords,)
    # cube_object_percent_gdf.plot()
    # gpd2postSQL(cube_object_percent_gdf,table_name='cube_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # cube_object_percent_gdf=postSQL2gpd(table_name='cube_object_percent',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # cube_object_percent_gdf=visual_entropy(cube_object_percent_gdf)   
    # gpd2postSQL(cube_object_percent_gdf,table_name='cube_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # cube_object_percent_gdf=idx_clustering(cube_object_percent_gdf,field='equilibrium_degree',n_clusters=10)
    # gpd2postSQL(cube_object_percent_gdf,table_name='cube_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    # cube_object_percent_gdf=postSQL2gpd(table_name='cube_object_percent',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # cube_object_percent_ground_gdf=cube_object_percent_gdf.copy(deep=True)
    # cube_object_percent_ground_gdf['ground']=cube_object_percent_ground_gdf.apply(lambda row:100-row.sky-row.vege-row.building-row.wall-row.fence-row.bboard,axis=1)
    # gpd2postSQL(cube_object_percent_ground_gdf,table_name='cube_object_ground_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')


    #frequency table by range    
    # bins=[0,15,25,50,100]    
    # cube_object_percent_gdf['sky_vege']=cube_object_percent_gdf.apply(lambda row:row.sky+row.vege,axis=1)
    # gpd2postSQL(cube_object_percent_gdf,table_name='cube_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
        
    # columns=['vege','sky','sky_vege','ground','equilibrium_degree',]
    # frequency=cube_object_percent_gdf[columns].apply(pd.Series.value_counts,bins=bins,)
    # for column_name in columns:
    #     frequency[column_name+'_percentage']=(frequency[column_name] / frequency[column_name].sum()) * 100
    # percentage=frequency[frequency.columns[-5:]]
    # percentage.to_excel('./graph/objects_percentage.xlsx')
    #__________________________________________________________________________
    pass


