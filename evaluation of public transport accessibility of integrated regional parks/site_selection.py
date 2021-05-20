# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:05:42 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
from database import postSQL2gpd,gpd2postSQL
import pickle
import networkx as nx
import libpysal
import inequality

nanjing_epsg=32650 #Nanjing 

def grid_pts(polygon,cell_size,epsg):
    import numpy as np
    import shapely
    import geopandas as gpd
    from tqdm import tqdm
    
    xmin, ymin, xmax, ymax=polygon.bounds
    print("xmin={}, ymin={}, xmax={}, ymax={}".format(xmin, ymin, xmax, ymax))
    grid_cells=[]
    for x0 in tqdm(np.arange(xmin, xmax+cell_size, cell_size )):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1=x0-cell_size
            y1=y0+cell_size
            box=shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(box.centroid) 
            # print(box.centroid)            
            # break
        # break
    cell_pts=gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:{}".format(epsg))    
    # print(cell_pts)
    return cell_pts

def cell_pts_shortest_path(G,cell_pts):
    from tqdm import tqdm
    import pandas as pd
    from shapely.geometry import Point,MultiPoint
    from shapely.ops import nearest_points
      
    update_shortest_length_dict={}
    # update_shortest_path_dict={}
    cell_pt2station_map={}
    
    stations_position={k:Point(v) for k,v in  nx.get_node_attributes(G,'position').items()}
    stations_position_pd=pd.DataFrame.from_dict(stations_position,orient='index',columns=['geometry'])   

    with tqdm(total=len(cell_pts)) as progress_bar:
        for idx,row in tqdm(cell_pts.iterrows()):
            # print(idx)
            cell_pt=row.geometry
            nearest_station=nearest_points(MultiPoint(stations_position_pd.geometry),cell_pt)[0]
            # print(nearest_station)
            nearest_station_PointiUid=stations_position_pd[stations_position_pd.geometry==nearest_station].index[0]
            # print(nearest_station_PointiUid)        
            
            # shortest_path=nx.shortest_path(G, source=nearest_station_PointiUid,weight="time_cost")
            shortest_length=nx.shortest_path_length(G, source=nearest_station_PointiUid,weight="time_cost")
            
            key_n=str(idx)+'_'+nearest_station_PointiUid
            # print(key_n)
            update_shortest_length_dict[key_n]=shortest_length
            # update_shortest_path_dict[key_n]=shortest_path
            cell_pt2station_map[key_n]=cell_pt
            
            progress_bar.update(1) # update progress
            # break
    return update_shortest_length_dict,cell_pt2station_map #update_shortest_path_dict,

    
def Gini_update(park_shortest_path,new_park_shortest_path,G_SB,epsg,col_name='mean'):
    import pandas as pd 
    from copy import deepcopy
    import geopandas as gpd
    import inequality
    from tqdm import tqdm
    from shapely.geometry import Point
    
    stations_pts={k:Point(v) for k,v in nx.get_node_attributes(G_SB,"position").items()}
    existed_park_sp={k:v[['stations','time_cost']] for k,v in park_shortest_path.items()}
    # print(len(existed_park_sp))
    new_park_sp={k:pd.DataFrame.from_dict(v,orient='index').reset_index().rename(columns={'index':'stations',0:'time_cost'}) for k,v in new_park_shortest_path.items()}
    # print(new_park_sp)
    gini_updated_dict={}
    for k,v in tqdm(new_park_sp.items()):
        existed_park_sp_deepcopy=deepcopy(existed_park_sp)
        # print(k,v)
        existed_park_sp_deepcopy[k]=v
        # print('\nupdated num=',len(existed_park_sp),len(existed_park_sp_deepcopy))
        update_time_cost=pd.concat([v[['stations','time_cost']].set_index(['stations']).rename(columns={'time_cost':k}) for k,v in existed_park_sp_deepcopy.items()],axis=1)
        update_tc_stats=update_time_cost.T.describe().T
        # update_tc_stats['geometry']=update_tc_stats.apply(lambda row:stations_pts[row.name] ,axis=1) 
        # update_tc_stats_gdf=gpd.GeoDataFrame(update_tc_stats,crs="EPSG:{}".format(epsg))
        
        # print(station2park_tc_stats_gdf)
        gini_updated=inequality.gini.Gini(update_tc_stats[col_name]).g
        print('\nGini=',gini_updated)
        # break
        gini_updated_dict[k]=gini_updated
        print(gini_updated_dict)
    with open('./processed data/gini_updated.pkl','wb') as f:
        pickle.dump(gini_updated_dict,f)
        
    return gini_updated_dict

def Gini_update_pool(park_shortest_path,new_park_shortest_path,G_SB,epsg,col_name='mean'):
    import pandas as pd 
    from copy import deepcopy
    import geopandas as gpd
    import inequality
    from tqdm import tqdm
    from shapely.geometry import Point
    from multiprocessing import Pool
    from functools import partial
    import site_selection_pool
    
    stations_pts={k:Point(v) for k,v in nx.get_node_attributes(G_SB,"position").items()}
    existed_park_sp={k:v[['stations','time_cost']] for k,v in park_shortest_path.items()}
    # print(len(existed_park_sp))
    new_park_sp={k:pd.DataFrame.from_dict(v,orient='index').reset_index().rename(columns={'index':'stations',0:'time_cost'}) for k,v in new_park_shortest_path.items()}
    # print(new_park_sp)
    
    k_v_list=[[k,v] for k,v in tqdm(new_park_sp.items())] #[:5]
    prod_args=partial(site_selection_pool.Gini, args=[existed_park_sp,col_name])
    with Pool(8) as p:
        gini_updated_list=p.map(prod_args,tqdm(k_v_list))        

    gini_updated_dict=dict(gini_updated_list)
    print(gini_updated_dict)        
        
    with open('./processed data/gini_updated.pkl','wb') as f:
        pickle.dump(gini_updated_dict,f)
        
    return gini_updated_dict

def Gini2gpd(gini_updated_dict,cell_pt2station_map,epsg):
    import pandas as pd
    import geopandas as gpd
    
    gini_updated_df=pd.DataFrame.from_dict(gini_updated_dict,orient='index',columns=['gini'])
    gini_updated_df['geometry']=gini_updated_df.index.map(cell_pt2station_map)
    # print(gini_updated_df)
    gini_updated_gdf=gpd.GeoDataFrame(gini_updated_df,crs="EPSG:{}".format(epsg))
    # print(gini_updated_gdf)
    return gini_updated_gdf
    

if __name__=="__main__":
    # G_SB=nx.read_gpickle("./network/G_SB.gpickle") 
    # station2park_tc_stats_gdf=postSQL2gpd(table_name='station2park_tc_stats',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    region=postSQL2gpd(table_name='region',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')    

    cell_pts=grid_pts(region.geometry[0],2000,nanjing_epsg)
    cell_pts.plot()
    # print('\n cell_pts num=',cell_pts.shape)
    # update_shortest_length_dict,cell_pt2station_map=cell_pts_shortest_path(G_SB,cell_pts)   #update_shortest_path_dict,  
    # with open('./processed data/update_shortest_length_dict.pkl','wb') as f:
    #     pickle.dump(update_shortest_length_dict,f)    
    # # with open('./processed data/update_shortest_path_dict.pkl','wb') as f:
    # #     pickle.dump(update_shortest_path_dict,f)  
    # with open('./processed data/cell_pt2station_map.pkl','wb') as f:
    #     pickle.dump(cell_pt2station_map,f)   

    # with open('./processed data/update_shortest_length_dict.pkl','rb') as f:
    #     update_shortest_length_dict=pickle.load(f)
    # with open('./processed data/cell_pt2station_map.pkl','rb') as f:
    #     cell_pt2station_map=pickle.load(f)        
        
    # with open('./processed data/park_shortest_path.pkl','rb') as f:
    #     park_shortest_path=pickle.load(f)
    # gini_updated_dict=Gini_update_pool(park_shortest_path,update_shortest_length_dict,G_SB,nanjing_epsg,'mean')  
    
    # with open('./processed data/gini_updated.pkl','rb') as f:
    #     gini_updated_dict=pickle.load(f)      
    # gini_updated_gdf=Gini2gpd(gini_updated_dict,cell_pt2station_map,nanjing_epsg)    
    # gini_updated_gdf.plot(column='gini')
    
    # gpd2postSQL(gini_updated_gdf,table_name='gini_updated',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')











