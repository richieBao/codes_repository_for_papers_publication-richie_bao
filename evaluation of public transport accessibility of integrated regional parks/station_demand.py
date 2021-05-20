# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:58:50 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
import pickle
import networkx as nx
from database import postSQL2gpd,gpd2postSQL


from segregation.aspatial import GiniSeg

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]    
nanjing_epsg=32650 #Nanjing 

def station_demand(park_shortest_path,G_SB,epsg):
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    
    stations_pts={k:Point(v) for k,v in nx.get_node_attributes(G_SB,"position").items()}
    # print(stations_pts)
    station2park_time_cost=pd.concat([v[['stations','time_cost']].set_index(['stations']).rename(columns={'time_cost':k}) for k,v in park_shortest_path.items()],axis=1)
    # print(station2park_time_cost.T)
    station2park_tc_stats=station2park_time_cost.T.describe().T
    # print(station2park_tc_stats)
    station2park_tc_stats['geometry']=station2park_tc_stats.apply(lambda row:stations_pts[row.name] ,axis=1)
    # print(station2park_tc_stats)
    station2park_tc_stats_gdf=gpd.GeoDataFrame(station2park_tc_stats,crs="EPSG:{}".format(epsg))
    return station2park_tc_stats_gdf

def station_demand_SDR(park_shortest_path,G_SB,population_weighted,epsg): # SDR-supply Demand Ratio
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    import statistics
    from scipy.stats import norm
    from tqdm import tqdm
    tqdm.pandas()
    
    stations_pts={k:Point(v) for k,v in nx.get_node_attributes(G_SB,"position").items()}
    # print(stations_pts)
    station2park_time_cost=pd.concat([v[['stations','time_cost']].set_index(['stations']).rename(columns={'time_cost':k}) for k,v in park_shortest_path.items()],axis=1)
    # print(station2park_time_cost)
    
    time_cost=flatten_lst([v.time_cost.to_list() for v in park_shortest_path.values()])
    tc_max=max(time_cost)
    tc_min=min(time_cost)
    tc_mean=statistics.mean(time_cost)
    tc_std=statistics.stdev(time_cost)   
    
    station2park_tc_stats=station2park_time_cost.T.describe().T
    # print(station2park_tc_stats)
    station2park_tc_stats['geometry']=station2park_tc_stats.progress_apply(lambda row:stations_pts[row.name] ,axis=1)
    # print(station2park_tc_stats)
    
    population_weighted['area']=population_weighted.geometry.apply(lambda row:row.area)
    populationWeighted=population_weighted[['Name_EN','populationWeighted','area']].set_index('Name_EN')
    populationWeighted_dict=populationWeighted.to_dict()#['populationWeighted']    
    # print(populationWeighted_dict)    
    
    station2park_tc_stats['SDR']=station2park_time_cost.progress_apply(lambda row:sum(
        [(1-norm.cdf(row[n],loc=tc_mean,scale=tc_std))*(populationWeighted_dict['area'][n.split('_')[-1]]/populationWeighted_dict['populationWeighted'][n.split('_')[-1]]) for n in station2park_time_cost.columns])
        ,axis=1)  
    station2park_tc_stats_gdf=gpd.GeoDataFrame(station2park_tc_stats,crs="EPSG:{}".format(epsg))    
    return station2park_tc_stats_gdf  
# station_demand_SDR(park_shortest_path,G_SB,comprehensive_park_en_withPopulationWeighted,nanjing_epsg)

def Gini_LorenzeCurve_singleVariable(df,col_name):
    import inequality
    import libpysal
    import quantecon as qe
    import matplotlib.pyplot as plt
    
    station2park_tc_mean_gini=inequality.gini.Gini(df[col_name])
    print('Gini={}'.format(station2park_tc_mean_gini.g))    
    
    f_vals, l_vals=qe.lorenz_curve(df[col_name].to_numpy()) # Cumulative share of people for each person index (i/n);Cumulative share of income for each person index
    # print(f_vals, l_vals)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(f_vals, l_vals, label='Lorenz curve, time cost mean')
    ax.plot(f_vals, f_vals, label='Lorenz curve, equality')
    ax.legend()
    plt.xlabel("Percentage of bus stations %",fontsize=14)
    plt.ylabel("Percentage of time cost %",fontsize=14)
    plt.savefig('./graph/Lorenze_curve.png')
    plt.show()    
    
def Gini_Spatial_LorenzeCurve_singleVariable(df_pts,df_polygons,col_name):
    import libpysal
    import numpy as np
    import inequality
    from tqdm import tqdm
    tqdm.pandas()
    
    def apply_func(pt):
        df_polygons['within']=df_polygons.geometry.apply(lambda row:pt.within(row))
        # print(df_polygons['within'])
        if df_polygons['within'].sum()>0:
            countryname=df_polygons[df_polygons['within']].countyname_EN.values[0]
        else:
            countryname='none'
        # print(countryname)
        return countryname
    
    df_pts['countyname']=df_pts.geometry.progress_apply(apply_func)
    print(df_pts)    
    w=libpysal.weights.block_weights(df_pts['countyname'])
    gs=inequality.gini.Gini_Spatial(df_pts[col_name],w)
    print('p_values={}'.format(gs.p_sim))
    print('Gini_Spatial={}'.format(gs.wcg_share))
    
    return df_pts

if __name__=="__main__":
    # with open('./processed data/population_flow_dict.pkl','rb') as f:
    #     population_flow_dict=pickle.load(f)        
    with open('./processed data/park_shortest_path.pkl','rb') as f:
        park_shortest_path=pickle.load(f)
    G_SB=nx.read_gpickle("./network/G_SB.gpickle") 

    # station2park_tc_stats_gdf=station_demand(park_shortest_path,G_SB,nanjing_epsg)
    # gpd2postSQL(station2park_tc_stats_gdf,table_name='station2park_tc_stats',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility') #Only lowercase fields are accepted.
         
    comprehensive_park_en_withPopulationWeighted=postSQL2gpd(table_name='comprehensive_park_en',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    station2park_tc_stats_SDR_gdf=station_demand_SDR(park_shortest_path,G_SB,comprehensive_park_en_withPopulationWeighted,nanjing_epsg)
    gpd2postSQL(station2park_tc_stats_SDR_gdf,table_name='station2park_sdr_stats',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility') #Only lowercase fields are accepted.
    Gini_LorenzeCurve_singleVariable(station2park_tc_stats_SDR_gdf,'SDR')    
    
    
    # station2park_tc_stats_gdf=postSQL2gpd(table_name='station2park_tc_stats',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    # Gini_LorenzeCurve_singleVariable(station2park_tc_stats_gdf,'mean')
    
    administrative_districts_EN=postSQL2gpd(table_name='admin_distr_en',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    # Gini_Spatial_gdf=Gini_Spatial_LorenzeCurve_singleVariable(station2park_tc_stats_gdf,administrative_districts_EN,'mean')  
    
    Gini_Spatial_LorenzeCurve_singleVariable(station2park_tc_stats_SDR_gdf,administrative_districts_EN,'SDR')
