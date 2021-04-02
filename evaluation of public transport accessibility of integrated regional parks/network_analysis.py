# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:24:36 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
import networkx as nx
from database import postSQL2gpd
import pickle
import joblib

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]    
def bus_shortest_paths(G,start_stops_PointUid_sp):
    from tqdm import tqdm
    
    all_shortest_length_dict={}
    all_shortest_path_dict={}
    
    for stop in tqdm(start_stops_PointUid_sp):
        shortest_path=nx.shortest_path(G, source=stop,weight="time_cost")
        print(shortest_path)
        shortest_length=nx.shortest_path_length(G, source=stop,weight="time_cost")
        all_shortest_length_dict[stop]=shortest_length
        all_shortest_path_dict[stop]=shortest_path
        break

    return all_shortest_path_dict,all_shortest_length_dict


if __name__=="__main__":
    comprehensivePark_adjacentStations=postSQL2gpd(table_name='adjacent_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    start_stops_PointUid=flatten_lst([eval(lst) for lst in comprehensivePark_adjacentStations.adjacent_PointUid])

    #NETWORK ANALYSIS
    G_SB=nx.read_gpickle("./network/G_SB.gpickle")
    
    all_shortest_path_dict_,all_shortest_length_dict_=bus_shortest_paths(G_SB,start_stops_PointUid)
    # shortest_routes_fre_df_,shortest_df_dict_=bus_service_index(G_SB,all_shortest_path_dict_,all_shortest_length_dict_,SES=[0,15,5])  #SES=[0,60*3,5]
    # with open('./processed data/all_shortest_path_dict.joblib.bz2','wb') as f:
    #     joblib.dump(all_shortest_path_dict_,f,compress=('bz2', 3))
    # with open('./processed data/all_shortest_length_dict.joblib.bz2','wb') as f:
    #     joblib.dump(all_shortest_length_dict_,f,compress=('bz2', 3))    
    
    