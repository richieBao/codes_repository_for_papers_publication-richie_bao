# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:35:59 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
import geopandas as gpd
import fiona,io
from tqdm import tqdm
import pyproj
import pandas as pd

nanjing_epsg=32650 #Nanjing
data_dic={
    'bus_routes':r'./data/bus route and station/bus_routes.shp',
    'bus_stations':r'./data/bus route and station/bus_stations.shp',
    'subway_lines':r'./data/subway station and line/subway_lines.shp',
    'subway_stations':r'./data/subway station and line/subway_stations.shp',
    'population':r'./data/population/population.shp',
    'comprehensive_park':r'./data/NanjingParks.kml',
    'region':r'./data/region/region.shp'
    }

def gpd2postSQL(gdf,table_name,**kwargs):
    from sqlalchemy import create_engine
    # engine=create_engine("postgres://postgres:123456@localhost:5432/workshop-LA-UP_IIT")  
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('has been written to into the PostSQL database...')
    
def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf    

def boundary_buffer(kml_extent,proj_epsg,bounadry_type='buffer_circle',buffer_distance=1000,**kwargs):
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Point,LinearRing,Polygon
    import geopandas as gpd
    import os
    
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    boundary_gdf=gpd.read_file(kml_extent,driver='KML')
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=pyproj.CRS('EPSG:{}'.format(proj_epsg))
    project=pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    boundary_proj=transform(project,boundary_gdf.geometry.values[0])    
    
    if bounadry_type=='buffer_circle':
        b_centroid=boundary_proj.centroid
        b_centroid_buffer=b_centroid.buffer(buffer_distance)
        c_area=[b_centroid_buffer.area]        
        b_centroid_gpd=gpd.GeoDataFrame({'x':[b_centroid.x],'y':[b_centroid.y],'geometry':[b_centroid]},crs=utm)
        # gpd2postSQL(b_centroid_gpd,table_name=kwargs['tablename'],myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase'])
        return b_centroid_buffer
    
    elif bounadry_type=='buffer_offset':
        boundary_=Polygon(boundary_proj.exterior.coords)
        LR_buffer=boundary_.buffer(buffer_distance,join_style=1).difference(boundary_)
        LR_area=[LR_buffer.area]
        gpd.GeoDataFrame({'area': LR_area,'geometry':LR_buffer},crs=utm).to_crs(wgs84).to_file('./data/GIS/LR_buffer.shp')  
        b_buffer_gpd=gpd.GeoDataFrame({'name':'boundary_buffer','geometry':[b_centroid]},crs=utm)
        # gpd2postSQL(b_buffer_gpd,table_name=kwargs['tablename'],myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase'])
        return LR_buffer

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

def parkName_translation(comprehensive_park):
    comprehensive_park.Name=comprehensive_park.Name.apply(lambda row:row.strip())
    name_mapping={
        '八字山公园':'Bazi Mountain',
        '北崮山公园':'Beigu Mountain',
        '大桥公园':'Bridge park',
        '古林公园':'Ancient Forest',
        '鼓楼公园':'Drum Tower',
        '老虎山公园':'Tiger Mountain',
        '幕府山公园':'Mufu Mountain',
        '清凉山公园':'Qingliang Mountain',
        '狮子山公园':'Lion Rock Mountain',
        '石头城公园':'Stone City Park',
        '乌龙潭公园':'Wulong Pool',
        '绣球公园':'Xiuqing Park',
        '河西中央公园':'Hexi Central Park',
        '绿博园':'Green EXPO Garden',
        '滨江公园':'Riverside Park',
        '莫愁湖公园':'Mochou Lake',
        '南湖公园':'South Lake',
        '百家湖公园':'Baijia Lake',
        '凤凰公园':'Phoenix Park',
        '九龙湖公园':'Jiulong Park',
        '竹山公园':'Zhushan Park',
        '九龙公园':'Jiulong Park',
        '六合凤凰山公园':'Liuhe Phoenix Mountain',
        '龙池公园':'Dragon Pool',
        '平顶山公园':'Pingding Mountain',
        '太子山公园':'Prince Mountain',
        '宝塔山公园':'Pagoda Hill',
        '北堡公园':'North Fort',
        '凤凰山公园':'Phoenix Mountain',
        '浦口公园':'Pukou Park',
        '二桥公园':'Two Bridges Park',
        '南炼公园':'Nanlian Park',
        '三叶湖公园':'Three Leaf Lake',
        '太平山公园':'Taiping Hill',
        '乌龙山公园':'Wulong Mountain',
        '燕子矶公园':'Swallow Rock',
        '白鹭洲公园':'Egret Island',
        '七桥瓮公园':'Qiqiaoweng Park',
        '午朝门公园':'Wuchaomen Park',
        '月牙湖公园':'Crescent Lake',
        '郑和公园':'Zhenghe Park',
        '白马公园':'White House Park',
        '北极阁公园':'Arctic Pavilion',
        '九华山公园':'Jiuhua Mountain',
        '聚宝山公园':'Treasure Hill',
        '梅花谷公园':'Plum Blossom Valley',
        '情侣园':'Couples Garden',
        '体育学院南公园':'South Park of Sports Institute',
        '玄武湖公园':'Xuanwu Lake',
        '花神湖公园':'Flora Lake',
        '菊花台公园':'Chrysanthemums Terrace',
        '莲花湖公园':'Lotus Lake',
        '梅山公园':'Plum Blossom Hill',          
        }
    comprehensive_park['Name_EN']=comprehensive_park['Name'].map(name_mapping)
    return comprehensive_park

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

def adjacent_stations(comprehensive_park,stations,buffer_distance=200):
    from shapely.geometry import MultiPoint,Polygon,Point
    from copy import deepcopy
    import geopandas as gpd
    
    comprehensivePark_adjacentStations=deepcopy(comprehensive_park)
    
    def adjacent_stations_single_multipoint(geo_polygon,return_type='mp'):
        buffer=geo_polygon.buffer(buffer_distance,join_style=1)
        stations['within']=stations.geometry.apply(lambda g:g.within(buffer)) 
        stations_within=stations[stations['within'].values]
                
        if return_type=='mp':
            return MultiPoint(stations_within['geometry'].to_list())
        elif return_type=='pu':
            return stations_within.PointUid.to_list()
        
    comprehensivePark_adjacentStations['adjacent_stations']=comprehensivePark_adjacentStations.geometry.apply(adjacent_stations_single_multipoint,args=('mp',))
    comprehensivePark_adjacentStations['adjacent_PointUid']=comprehensivePark_adjacentStations.geometry.apply(adjacent_stations_single_multipoint,args=('pu',))
    comprehensivePark_adjacentStations.rename(columns={'geometry':'geo_park','adjacent_stations':'geometry'},inplace=True)   
    comprehensivePark_adjacentStations['adjacent_num']=comprehensivePark_adjacentStations.adjacent_PointUid.apply(lambda row:len(row))
    comprehensivePark_adjacentStations['park_perimeter']=comprehensivePark_adjacentStations.apply(lambda row:row.geo_park.length,axis=1)
    comprehensivePark_adjacentStations['park_area']=comprehensivePark_adjacentStations.apply(lambda row:row.geo_park.area,axis=1)
    comprehensivePark_adjacentStations['adjacent_perimeterRatio']=comprehensivePark_adjacentStations.apply(lambda row:len(row.adjacent_PointUid)/row.geo_park.length,axis=1)
    comprehensivePark_adjacentStations['adjacent_areaRatio']=comprehensivePark_adjacentStations.apply(lambda row:len(row.adjacent_PointUid)/row.geo_park.area,axis=1)    
    
    comprehensivePark_buffer=deepcopy(comprehensivePark_adjacentStations)
    comprehensivePark_buffer['buffer']=comprehensivePark_buffer.geo_park.apply(lambda row:row.buffer(buffer_distance,join_style=1))
    comprehensivePark_buffer.drop(['geometry','geo_park'],axis=1,inplace=True)
    comprehensivePark_buffer.rename(columns={'buffer':'geometry'},inplace=True)
    
    
    comprehensivePark_adjacentStations.drop(['geo_park'],axis=1,inplace=True)
    return comprehensivePark_adjacentStations,comprehensivePark_buffer

if __name__=="__main__":
    #region
    region=shp2gdf(data_dic['region'],epsg=nanjing_epsg,boundary=None,encoding='GBK')
    region.plot()   
    # gpd2postSQL(region,table_name='region',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')    
    
    '''
    #comprehensive park
    comprehensive_park=kml2gdf_folder(data_dic['comprehensive_park'],epsg=nanjing_epsg,boundary=None) 
    gpd2postSQL(comprehensive_park,table_name='comprehensive_park',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    
    # #CH->EN
    comprehensive_park_EN=parkName_translation(comprehensive_park)
    gpd2postSQL(comprehensive_park_EN,table_name='comprehensive_park_en',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility') #Only lowercase fields are accepted.

    # #bus routes
    bus_routes=shp2gdf(data_dic['bus_routes'],epsg=nanjing_epsg,boundary=None,encoding='GBK')
    bus_routes.plot()
    gpd2postSQL(bus_routes,table_name='bus_routes',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')

    # #bus station
    bus_stations=shp2gdf(data_dic['bus_stations'],epsg=nanjing_epsg,boundary=region.geometry[0],encoding='GBK')
    bus_stations.plot()
    gpd2postSQL(bus_stations,table_name='bus_stations',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    
    # #subway lines
    subway_lines=shp2gdf(data_dic['subway_lines'],epsg=nanjing_epsg,boundary=None,encoding='GBK')
    subway_lines.plot()
    gpd2postSQL(subway_lines,table_name='subway_lines',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    
    # #subway stations
    subway_stations=shp2gdf(data_dic['subway_stations'],epsg=nanjing_epsg,boundary=region.geometry[0],encoding='GBK')
    subway_stations.plot()
    gpd2postSQL(subway_stations,table_name='subway_stations',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
        
    # get adjacent stations of each park
    columns=['PointUid','geometry']
    stations=pd.concat([bus_stations[columns],subway_stations[columns]],ignore_index=True)
    comprehensivePark_adjacentStations,comprehensivePark_buffer=adjacent_stations(comprehensive_park_EN,stations,buffer_distance=400) 
    comprehensivePark_adjacentStations.plot()
    gpd2postSQL(comprehensivePark_adjacentStations,table_name='adjacent_stations',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')

    comprehensivePark_buffer.plot()
    gpd2postSQL(comprehensivePark_buffer,table_name='park_buffer',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')

    #population
    # population=shp2gdf(data_dic['population'],epsg=nanjing_epsg,boundary=None,encoding='GBK')
    # population.plot(column='Population',cmap='hot')
    # gpd2postSQL(population,table_name='population',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')   
    '''