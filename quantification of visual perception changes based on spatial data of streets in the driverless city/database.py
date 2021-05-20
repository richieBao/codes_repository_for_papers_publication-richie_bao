# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:45:39 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import geopandas as gpd
import fiona,io
from tqdm import tqdm
import pyproj
import pandas as pd


#CREATE EXTENSION postgis;

xian_epsg=32649 #Xi'an   WGS84 / UTM zone 49N
data_dic={
    'roads':r'./data/Xian road/Xian road.shp',  
    'region_merge':r'./data/region/region_merge.shp',
    'region':r'./data/region/region.shp',    
    'old_city_boundary':'./data/old city/old city.kml',
    'tour_line':'./data/old city/tour line.kml',
    'tour_line_seg':'./data/region/tour_line_seg.shp'
    }

def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf   

def shp2gdf(fn,epsg=None,boundary=None,encoding='utf-8'):
    import geopandas as gpd
    
    shp_gdf=gpd.read_file(fn,encoding=encoding)
    print('original data info:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(how='all',axis=1,inplace=True)
    print('dropna-how=all,result:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(inplace=True)
    print('dropna-several rows,result:{}'.format(shp_gdf.shape))
    # print(shp_gdf)
    if epsg is not None:
        shp_gdf_proj=shp_gdf.to_crs(epsg=epsg)
    if boundary:
        shp_gdf_proj['mask']=shp_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        shp_gdf_proj.query('mask',inplace=True)        
    
    return shp_gdf_proj

def gpd2postSQL(gdf,table_name,**kwargs):
    from sqlalchemy import create_engine
    # engine=create_engine("postgres://postgres:123456@localhost:5432/workshop-LA-UP_IIT")  
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('has been written to into the PostSQL database...')

def kml2gdf_folder(fn,epsg=None,boundary=None): 
    import pandas as pd
    import geopandas as gpd
    import fiona,io

    # Enable fiona driver
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    kml_gdf=gpd.GeoDataFrame()
    for layer in tqdm(fiona.listlayers(fn)):
        # print("_"*50)
        # print(layer)
        src=fiona.open(fn, layer=layer)
        meta = src.meta
        meta['driver'] = 'KML'        
        with io.BytesIO() as buffer:
            with fiona.open(buffer, 'w', **meta) as dst:            
                for i, feature in enumerate(src):
                    # print(feature)
                    # print("_"*50)
                    # print(feature['geometry']['coordinates'])
                    if len(feature['geometry']['coordinates'][0]) > 1:
                        # print(feature['geometry']['coordinates'])
                        
                        dst.write(feature)
                        # break
            buffer.seek(0)
            one_layer=gpd.read_file(buffer,driver='KML')
            # print(one_layer)
            one_layer['group']=layer
            kml_gdf=kml_gdf.append(one_layer,ignore_index=True)

    if epsg is not None:
        kml_gdf_proj=kml_gdf.to_crs(epsg=epsg)

    if boundary:
        kml_gdf_proj['mask']=kml_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        kml_gdf_proj.query('mask',inplace=True)        

    return kml_gdf_proj    

if __name__=="__main__":
    pass
    #region_merge | boundary
    # region_merge=shp2gdf(data_dic['region_merge'],epsg=xian_epsg,boundary=None,encoding='GBK')
    # region_merge.plot()   
    # gpd2postSQL(region_merge,table_name='region_merge',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')       
    
    #region with name
    # region=shp2gdf(data_dic['region'],epsg=xian_epsg,boundary=None,encoding='GBK')
    # region.plot()
    # region_name_mapping={'bqq':'Baoqiao District','wyq':'Weiyang District','lhq':'Lianhu District','ytq':'Yanta District','xcq':'Xincheng District','blq':'Beilin District'}
    # region['name_en']=region['PYNAME'].map(region_name_mapping)
    # gpd2postSQL(region,table_name='region',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV') 
    
    #roads within designated boundary
    # roads=shp2gdf(data_dic['roads'],epsg=xian_epsg,boundary=region_merge.geometry[0],encoding='GBK')
    # roads.plot()
    # gpd2postSQL(roads,table_name='roads',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')   
    
    #old city
    # old_city_boundary=kml2gdf_folder(data_dic['old_city_boundary'],epsg=xian_epsg,boundary=None) 
    # gpd2postSQL(old_city_boundary,table_name='old_city_boundary',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')    
    # roads_oldCity=shp2gdf(data_dic['roads'],epsg=xian_epsg,boundary=old_city_boundary.geometry[0],encoding='GBK')
    # gpd2postSQL(roads_oldCity,table_name='roads_oldCity',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV') 
    # roads_oldCity.plot()
    
    #tour line tour_line
    # tour_line=kml2gdf_folder(data_dic['tour_line'],epsg=xian_epsg,boundary=None) 
    # gpd2postSQL(tour_line,table_name='tour_line',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV') 
    
    #tour_line_seg
    # tour_line_seg=shp2gdf(data_dic['tour_line_seg'],epsg=xian_epsg,boundary=None,encoding='GBK')
    # tour_line_seg.plot()
    # gpd2postSQL(tour_line_seg,table_name='tour_line_seg',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV') 
    
    #POI
    poi_gdf=postSQL2gpd(table_name='poi',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    poi_gdf['level_0_name']=poi_gdf.level_0.apply(lambda row:row.split("_")[-1])
    gpd2postSQL(poi_gdf,table_name='poi',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
