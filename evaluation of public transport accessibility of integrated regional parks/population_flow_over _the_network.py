# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:23:13 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
import networkx as nx
from database import postSQL2gpd,gpd2postSQL
import pickle,os,copy

import matplotlib.pyplot as plt
import pandas as pd
import pickle

plt.rcParams['font.sans-serif'] = ['DengXian'] # 指定默认字体 'KaiTi','SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

nanjing_epsg=32650 #Nanjing 

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]    
def population_flowOverNetwork(G_SB,population,park_shortest_path):
    from shapely.ops import nearest_points
    import pandas as pd
    from shapely.geometry import MultiPoint,Point
    from tqdm import tqdm
    import pickle
    import copy
    import statistics
    from scipy.stats import norm

    stations_pts={k:Point(v) for k,v in nx.get_node_attributes(G_SB,"position").items()}
    stations_mpts=MultiPoint(list(stations_pts.values()))
       
    time_cost=flatten_lst([v.time_cost.to_list() for v in park_shortest_path.values()])
    tc_max=max(time_cost)
    tc_min=min(time_cost)
    tc_mean=statistics.mean(time_cost)
    tc_std=statistics.stdev(time_cost)    
    
    population_flow_dict={}
    def row_func(row):
        nearest_pt=nearest_points(row,stations_mpts)[1]
        # nearest_pt_row=shortest_path_df[shortest_path_df["station_geometry"]==nearest_pt]
        nearest_pt_row=shortest_path_df[shortest_path_df["station_geometry"]==str(nearest_pt)]
 
        return [nearest_pt,nearest_pt_row.idx_adjacent.values[0],nearest_pt_row.time_cost.values[0],nearest_pt_row.stations.values[0]]
    
    for park,shortest_path_df in tqdm(park_shortest_path.items()):
        population_flow=copy.deepcopy(population)
        shortest_path_df['station_geometry']=shortest_path_df.stations.apply(lambda row:str(stations_pts[row]))   
        nearest_info=population_flow.geometry.apply(row_func)
        col_n=['nearest_pt','idx_adjacent','time_cost','stations']
        nearest_info_T=[list(i) for i in zip(*nearest_info.to_list())]        
        for n,info in zip(col_n,nearest_info_T):
            population_flow[n]=info
        
        # print(population_flow) 
        population_flow['tc_weight']=population_flow.time_cost.apply(lambda row:1-(row-tc_min)/(tc_max-tc_min))
        population_flow['Gaussian_weight']=population_flow.time_cost.apply(lambda row:1-norm.cdf(row,loc=tc_mean,scale=tc_std)) #norm.sf(x, loc=0, scale=1)=1-cdf
        population_flow['population_Gweighted']=population_flow.apply(lambda row:int(row.Population)*row.Gaussian_weight,axis=1)
        # break
        population_flow_dict[park]=population_flow
        
    with open('./processed data/population_flow_dict.pkl','wb') as f:
        pickle.dump(population_flow_dict,f)        
    return population_flow_dict

def populationWeighted_adjacentStations(population_flow_dict,G_SB,epsg,comprehensive_park_en):
    from tqdm import tqdm
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    
    stations_pts={k:Point(v) for k,v in nx.get_node_attributes(G_SB,"position").items()}
    populationWeighted_adjacentStations_dict={}
    for park,population_flow in tqdm(population_flow_dict.items()):
        populationGWeighted_group_sum=population_flow[['idx_adjacent','population_Gweighted']].groupby(by=['idx_adjacent']).sum()
        populationGWeighted_group_sum.reset_index(inplace=True)
        populationGWeighted_group_sum.rename(columns={'index':'idx_adjacent'})
        populationGWeighted_group_sum['geometry']=populationGWeighted_group_sum.idx_adjacent.apply(lambda row:stations_pts[row])
        populationWeighted_adjacentStations_dict[park]=populationGWeighted_group_sum        
        
    populationWeighted_adjacentStations_stack=pd.concat(populationWeighted_adjacentStations_dict.values(),keys=populationWeighted_adjacentStations_dict.keys())
    populationWeighted_adjacentStations_stack['populationWeighted_10thous']=populationWeighted_adjacentStations_stack.population_Gweighted.apply(lambda row:row/10000)
    # print(populationWeighted_adjacentStations_stack.columns)
    populationWeighted_adjacentStations_stack_gdf=gpd.GeoDataFrame(populationWeighted_adjacentStations_stack,crs="EPSG:{}".format(epsg))
    populationWeighted_parks={k.split("_")[-1]:v.population_Gweighted.sum() for k,v in populationWeighted_adjacentStations_dict.items()}    
    
    comprehensive_park_en['populationWeighted']=comprehensive_park_en['Name_EN'].map(populationWeighted_parks)  
    comprehensive_park_en['populationWeighted_10thous']=comprehensive_park_en.populationWeighted.apply(lambda row:row/10000)
    
    return populationWeighted_adjacentStations_dict,populationWeighted_adjacentStations_stack_gdf,comprehensive_park_en


if __name__=="__main__":
    # comprehensivePark_adjacentStations=postSQL2gpd(table_name='adjacent_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    population=postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    # G_SB=nx.read_gpickle("./network/G_SB.gpickle") 
    # with open('./processed data/park_shortest_path.pkl','rb') as f:
    #     park_shortest_path=pickle.load(f)
    # population_flow_dict=population_flowOverNetwork(G_SB,population,park_shortest_path)    
    
    # comprehensive_park_en=postSQL2gpd(table_name='comprehensive_park_en',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')    
    with open('./processed data/population_flow_dict.pkl','rb') as f:
        population_flow_dict=pickle.load(f)
    # populationWeighted_adjacentStations_dict,populationWeighted_adjacentStations_stack_gdf,comprehensive_park_en=populationWeighted_adjacentStations(population_flow_dict,G_SB,nanjing_epsg,comprehensive_park_en)  
    # gpd2postSQL(populationWeighted_adjacentStations_stack_gdf,table_name='adja_popu',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')   
    # gpd2postSQL(comprehensive_park_en,table_name='comprehensive_park_en',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility') #Only lowercase fields are accepted.

    
        
        
        
        
        