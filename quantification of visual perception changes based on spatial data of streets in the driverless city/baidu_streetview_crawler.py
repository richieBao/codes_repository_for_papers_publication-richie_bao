# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:46:58 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from database import postSQL2gpd,gpd2postSQL
import numpy as np
import pickle
import shutil

xian_epsg=32649 #Xi'an   WGS84 / UTM zone 49N
wgs84_epsg=4326

def roads_pts4bsv(roads_gdf,distance=10):
    from tqdm import tqdm
    import numpy as np
    from shapely.geometry import MultiPoint
    import pyproj
    from shapely.ops import transform
    import geopandas as gpd
    
    tqdm.pandas()    
    def line_pts(line):
        dists=np.arange(0,line.length,distance)
        pts=MultiPoint([line.interpolate(d,normalized=False) for d in dists])
        return pts      
        
    roads_gdf['pts']=roads_gdf.geometry.progress_apply(line_pts)
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=roads_gdf.crs #pyproj.CRS(roads_gpd.crs.srs)
    project=pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    roads_gdf['pts_wgs84']=roads_gdf.pts.progress_apply(lambda row:transform(project,row))    
    
    pts_gdf=gpd.GeoDataFrame(roads_gdf[['Name','Uid']],geometry=roads_gdf.pts_wgs84.to_list(),crs=wgs84) #roads_gdf.drop(['geometry'],axis=1)    
    return pts_gdf

def roads_pts4bsv_tourLine(roads_gdf,distance=10):
    from tqdm import tqdm
    import numpy as np
    from shapely.geometry import MultiPoint
    import pyproj
    from shapely.ops import transform
    import geopandas as gpd
    
    tqdm.pandas()    
    def line_pts(line):
        dists=np.arange(0,line.length,distance)
        pts=MultiPoint([line.interpolate(d,normalized=False) for d in dists])
        return pts      
        
    roads_gdf['pts']=roads_gdf.geometry.progress_apply(line_pts)
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=roads_gdf.crs #pyproj.CRS(roads_gpd.crs.srs)
    project=pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    roads_gdf['pts_wgs84']=roads_gdf.pts.progress_apply(lambda row:transform(project,row))    
    
    pts_gdf=gpd.GeoDataFrame(roads_gdf[['Name','group']],geometry=roads_gdf.pts_wgs84.to_list(),crs=wgs84) #roads_gdf.drop(['geometry'],axis=1)    
    return pts_gdf
    
   
def baidu_steetview_crawler(pts_gdf,save_path):
    import urllib,os
    from tqdm import tqdm
    import pickle
    
    downloadError_idx=[]
    coords={}
    pts_num={}    
    pt_fns={}
    for idx,row in pts_gdf.iterrows():
        pt_coords=[(pt.x,pt.y) for pt in row.geometry]
        coords[row.Name]=pt_coords
        pts_num[row.Name]=len(pt_coords)
    # print(coords)
    print("\npts_num={}".format(sum(pts_num.values())))
    
    urlRoot=r"http://api.map.baidu.com/panorama/v2?"
    query_dic={
        'width':'1024',
        'height':'512', 
        'fov':'360',
        'heading':'0',
        'pitch':'0',
        'coordtype':'wgs84ll',
        'ak':'rSxNX840wLxwVVhs5RDInfPqegZ12G78',
    }    
    
    # tt=0
    # for k,v in coords.items(): #for k,v in tqdm(coords.items()):
    for k,v in tqdm(coords.items()):
        # print(k,v)
        pt_fn=[]
        # for i,coord in enumerate(tqdm(v)): #for i,coord in enumerate(v): [522:]
        for i,coord in enumerate(v):
            # print(i,coord)
            # i=i+522
            pic_fn=os.path.join(save_path,"{}_{}.jpg".format(k,i))
                        
            if not os.path.exists(pic_fn):
                #update query arguments
                query_dic.update({
                                  'location':str(coord[0])+','+str(coord[1]),
                                 })         
                url=urlRoot+urllib.parse.urlencode(query_dic)
                # print(url)
                try:
                    data=urllib.request.urlopen(url)
                    pt_fn.append(pic_fn)
                    # print(data)
                    with open(pic_fn,'wb') as fp:
                        fp.write(data.read())           
                except:
                    downloadError_idx.append((k,i))
                    print('download_error:{},{}'.format(k,i))
            else:
                print("file existed.")
                
        pt_fns[k]=pt_fn

        # if tt==2:break
        # tt+=1
        
    with open('./processed data/pt_fns_tourLine.pkl','wb') as f:
        pickle.dump(pt_fns,f)
    with open('./processed data/coords_tourLine.pkl','wb') as f:
        pickle.dump(coords,f)       
    with open('./processed data/downloadError_idx_tourLine.pkl','wb') as f:
        pickle.dump(downloadError_idx,f)      
            
    return coords,pts_num

def img_valid(img_fns):
    from tqdm import tqdm
    import pickle
    
    img_val={}
    img_inval=[]
    for k,v in tqdm(img_fns.items()):
        fns=[]
        for fn in v:
            try:
                im=Image.open(fn)
                fns.append(fn)
            except:
                img_inval.append(fn)        
        img_val[k]=fns
            
    with open('./processed data/imgVal_fns.pkl','wb') as f:
        pickle.dump(img_val,f)        
    with open('./processed data/imgInval_fns.pkl','wb') as f:
        pickle.dump(img_inval,f)                 
    return img_val

def img_valid_copy_folder(imgs_root):
    from tqdm import tqdm
    import pickle
    import glob,os 
    from PIL import Image
    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))
    # print(img_fns)    
    
    img_val=[]
    img_inval=[]
    for fn in tqdm(img_fns):
        try:
            im=Image.open(fn)
            img_val.append(fn)
        except:
            img_inval.append(fn)    
    # print(img_val)
    for fn in tqdm(img_val):
        shutil.copy(fn,r'C:\Users\richi\omen_richiebao\omen_github\codes_repository_for_papers_publication-richie_bao\quantification of visual perception changes based on spatial data of streets in the driverless city\data\panoramic imgs_tour line valid')                

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

def pts_number_check(pts):
    downloadError_idx=[]
    coords={}
    pts_num={}    
    pt_fns={}
    for idx,row in pts.iterrows():
        pt_coords=[(pt.x,pt.y) for pt in row.geometry]
        coords[row.Name]=pt_coords
        pts_num[row.Name]=len(pt_coords)
    # print(coords)
    print("\npts_num={}".format(sum(pts_num.values())))


if __name__=="__main__":
    #A-4 regions of Xian
    # roads=postSQL2gpd(table_name='roads',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # # roads.plot()
    
    # distance=200  #200:pts_nun=14973
    # pts_gdf=roads_pts4bsv(roads,distance)
    # gpd2postSQL(pts_gdf,table_name='pts',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # save_path='./data/panoramic imgs'
    # coords,pts_num=baidu_steetview_crawler(pts_gdf,save_path)
 
    
    # with open('./processed data/pt_fns.pkl','rb') as f:
    #     pt_fns=pickle.load(f)
        
    # img_val=img_valid(pt_fns)  #14973 /13359  14973-13359=1614
    # for k,v in tqdm(img_val.items()):
    #     for fn in v:
    #         shutil.copy(fn,r'C:\Users\richi\omen_richiebao\omen_code\GSV\PASS-master\dataset\leftImg8bit\val\cs')
    
    #B-old city
    # roads_oldCity=postSQL2gpd(table_name='roads_oldCity',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # roads_oldCity.plot()
    # distance_oldCity=20  #200:pts_nun=14973
    # pts_oldCity_gdf=roads_pts4bsv(roads_oldCity,distance_oldCity)
    # pts_number_check(pts_oldCity_gdf)
    # gpd2postSQL(pts_oldCity_gdf,table_name='pts_oldCity_gdf',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')    
    # save_path='./data/panoramic imgs_old city'
    # coords_oldCity,pts_num_oldCity=baidu_steetview_crawler(pts_gdf,save_path)
    
    #C-tour line
    # tour_line=postSQL2gpd(table_name='tour_line',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # tour_line.plot()
    # distance_tourLine=8  #200:pts_nun=14973
    # pts_tourLine_gdf=roads_pts4bsv_tourLine(tour_line,distance_tourLine)
    # pts_tourLine_gdf.plot()
    # pts_number_check(pts_tourLine_gdf)
    # gpd2postSQL(pts_tourLine_gdf,table_name='pts_tourline',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV') 
    save_path='./data/panoramic imgs_tour line'
    # coords_oldCity,pts_num_oldCity=baidu_steetview_crawler(pts_tourLine_gdf,save_path)
    img_valid_copy_folder(save_path)
