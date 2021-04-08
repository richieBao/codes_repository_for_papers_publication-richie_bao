# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:24:36 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
import networkx as nx
from database import postSQL2gpd,gpd2postSQL
import pickle,os,copy

nanjing_epsg=32650 #Nanjing
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]    
def bus_shortest_paths(G,start_stops_PointUid_sp):
    from tqdm import tqdm
    
    all_shortest_length_dict={}
    all_shortest_path_dict={}
    
    for stop in tqdm(start_stops_PointUid_sp):
        shortest_path=nx.shortest_path(G, source=stop,weight="time_cost")
        # print(len(shortest_path.keys()))
        shortest_length=nx.shortest_path_length(G, source=stop,weight="time_cost")
        all_shortest_length_dict[stop]=shortest_length
        all_shortest_path_dict[stop]=shortest_path
        # break

    return all_shortest_path_dict,all_shortest_length_dict

def bus_shortest_paths_loop(adjacent_stations,G,save_root='processed data/shortest_path'):
    import pickle,os
    from tqdm import tqdm
    
    fn_lst=[]
    for idx,row in adjacent_stations.iterrows():
        # print(row.adjacent_PointUid)
        all_shortest_path_dict_,all_shortest_length_dict_=bus_shortest_paths(G,eval(row.adjacent_PointUid))
        fn=os.path.join(save_root,str(idx)+'_'+row.Name_EN+'.pkl')
        with open(fn,'wb') as f:
            pickle.dump({'path':all_shortest_path_dict_,'length':all_shortest_length_dict_},f)   
        fn_lst.append(fn)    
        print('\nsaved {}_{}'.format(idx,row.Name_EN))  
        # break
    
    with open(os.path.join(save_root,'fn_lst.pkl'),'wb') as f:
        pickle.dump(fn_lst,f)     
    
def park_station_shortestPaths(shortestPaths_fn_lst,epsg):
    import pandas as pd
    import numpy as np
    from shapely.geometry import LineString
    import geopandas as gpd
    from pathlib import Path
    from tqdm import tqdm
    
    station_position=nx.get_node_attributes(G_SB,'position')
    
    park_shortest_path={}
    for fn in tqdm(shortestPaths_fn_lst):
        # print(fn)
        with open(fn,'rb') as f:
            shortest_path_length=pickle.load(f)
        all_shortest_path_dict=shortest_path_length['path']
        all_shortest_length_dict=shortest_path_length['length']
        # print(all_shortest_path_dict[list(all_shortest_path_dict.keys())[0]])

        all_shortest_length_df=pd.DataFrame.from_dict(all_shortest_length_dict,orient='index')
        all_shortest_length_df.replace({0:np.NaN},inplace=True)
        min_length=all_shortest_length_df.min()
        # print(min_length.max())
        min_idx=all_shortest_length_df.idxmin()
        min_concat=pd.concat([min_idx,min_length],axis=1).reset_index().rename(columns={0:'idx_adjacent',1:"time_cost",'index':'stations'})      
        # print(min_concat.length.max())
        min_concat['path']=min_concat.apply(lambda row:all_shortest_path_dict[row.idx_adjacent][row.stations],axis=1)
        min_concat['geometry']=min_concat.path.apply(lambda row:LineString([station_position[PointUid] for PointUid in row]))
        # print(min_concat.time_cost)
        min_concat_gdf=gpd.GeoDataFrame(min_concat,geometry='geometry',crs=epsg)
        # print(min_concat_gdf.time_cost)
        park_shortest_path[Path(fn).stem]=min_concat_gdf
        
        # break
    with open('./processed data/park_shortest_path.pkl','wb') as f:
        pickle.dump(park_shortest_path,f)          
    # return park_shortest_path
    
def stats_shortest_length(park_shortest_path,comprehensive_park_en):
    import pandas as pd
    import matplotlib.pyplot as plt
    import geopandas as gpd
    
    shortest_timeCost=pd.concat({k:v.time_cost for k,v in park_shortest_path.items()},axis=1)     
    xticklabels_name={name:name.split('_')[-1] for name in shortest_timeCost.columns}
    shortest_timeCost.rename(columns=xticklabels_name,inplace=True)
    plt.rcParams["font.family"]='Arial' #Arial;"Times New Roman"
    plt.rcParams["font.size"]="40"
    color = {
            "boxes": "DarkGreen",
          "whiskers": "DarkOrange",
          "medians": "DarkBlue",
              "caps": "Gray",
    }    
    ax=shortest_timeCost.plot.box(sym="k+",showmeans=True,meanline=False,showfliers=False,figsize=(30*2,10*2)) #color=color, ;https://matplotlib.org/2.0.2/api/colors_api.html
    ax.set_xticklabels(shortest_timeCost.columns,rotation=90)
    ax.set_ylabel("Time Cost (min)")
    # plt.savefig('./graph/park_shortestPath_timecost.png')
    stats=shortest_timeCost.describe()
    stats.to_excel('./graph/park_shortestPath_timecost.xlsx',sheet_name='park_shortestPath_timecost')
    print(stats)
    
    comprehensive_park_en_timecost=gpd.GeoDataFrame(pd.merge(stats.T.rename(columns={n:'timecost_{}'.format(n) for n in stats.T.columns}),comprehensive_park_en,left_index=True,right_on='Name_EN'),crs=comprehensive_park_en.crs)
      
    return stats,comprehensive_park_en_timecost

def G_draw_paths_composite(G,range_paths,adjacent_PointUid,title_name,figsize=(30, 30),level=3):
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    from collections import defaultdict
    import matplotlib
    import numpy as np
    
    fig, ax=plt.subplots(figsize=figsize)
    ax.set_title(title_name,fontsize=300)
    
    range_paths.dropna(inplace=True)
    ranges=range_paths.range.unique()[:level]
    print('\nranges:',ranges)
    path_edges_dict={}
    for i in ranges:
        range_paths_range=range_paths[range_paths.range==i]
        path_edges=range_paths_range.path.to_list()    
        path_edges_dict[i]=path_edges
    
    G_p=G.copy()
    G_p.remove_edges_from(list(G_p.edges()))
    G_p_range=nx.Graph(G_p.subgraph(flatten_lst(list(path_edges_dict.values()))))
    
    edges_dict=defaultdict(list)
    for k in path_edges_dict.keys():
        for r in path_edges_dict[k]:
            path_edges=[(r[n],r[n+1]) for n in range(len(r)-1)]
            G_p_range.add_edges_from(path_edges)
            edges_dict[k].append(path_edges)
    print("Graph has %d nodes with %d paths(edges)" %(G_p_range.number_of_nodes(), G_p_range.number_of_edges()))   
        
    
    pos=nx.get_node_attributes(G_p_range,'position')
    nx.draw(G_p_range,pos=pos,node_size=100,ax=ax)

    G_start_stops=G_p_range.subgraph(adjacent_PointUid)
    pos_adjacent=nx.get_node_attributes(G_start_stops,'position')
    nx.draw_networkx_nodes(G_start_stops,pos=pos_adjacent,node_size=1000*2,ax=ax,node_color='red')
    # nx.draw_networkx_nodes(G_p_range,pos=pos,node_size=10) #,with_labels=True,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=20
    # nx.draw_networkx_labels(G_p_range,pos=pos,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=8)
    linewidths=[5+i*5 for i in range(len(ranges))] #[5,10,15,20,25]
    linewidths.reverse()
    cmap=matplotlib.cm.get_cmap('tab20b') #'tab20b'
    colors=[cmap(i) for i in np.linspace(0,1,len(ranges))]
    for j,k in enumerate(reversed(list(edges_dict.keys()))):
        # color=(random.random(), random.random(),random.random())
        for i, edgelist in enumerate(edges_dict[k]):            
            nx.draw_networkx_edges(G_p_range,pos=pos,edgelist=edgelist,edge_color=colors[j],width=linewidths[j],ax=ax) #edge_color = colors[ctr], width=linewidths[ctr]
    plt.savefig('./graph/level/{}.png'.format(title_name))    
    
    return edges_dict

def service_area_levels(G_SB,park_shortest_path,comprehensivePark_adjacentStations,SES=[0,25,5],figsize=(30, 30),level=3):
    import pandas as pd
    import numpy as np
    from shapely.geometry import LineString
    import geopandas as gpd
    from pathlib import Path
    from tqdm import tqdm

    start,end,step=SES      #start,end,step=0,10000,2000
    def partition(value,start,end,step):
        import numpy as np
        ranges_list=list(range(start,end+step,step))
        ranges=[(ranges_list[i],ranges_list[i+1]) for i in  range(len(ranges_list)-1)]
        for r in ranges:
            # print(r,value)
            if r[0]<=value<r[1] :    
                return r[1]           
        
    park_serviceArea_level={}
    park_serviceArea_level_fre={}
    for park,shortest_path in tqdm(park_shortest_path.items()):
        shortest_path['range']=shortest_path.time_cost.apply(partition,args=(start,end,step))
        ranges_frequency=shortest_path['range'].value_counts()
        park_serviceArea_level_fre[park]=ranges_frequency.to_dict()        
        park_serviceArea_level[park]=shortest_path
        park_serviceArea_level_fre_df=pd.DataFrame.from_dict(park_serviceArea_level_fre,orient='index')        
        # print(shortest_path.range.unique())
        park_name=park.split('_')[-1]
        adjacent_PointUid=eval(comprehensivePark_adjacentStations[comprehensivePark_adjacentStations.Name_EN==park_name].adjacent_PointUid.to_list()[0])
        edges_dict_=G_draw_paths_composite(G_SB,shortest_path,adjacent_PointUid,figsize=figsize,level=level,title_name=park_name) 
        
        # break
    park_serviceArea_level_fre_df=pd.DataFrame.from_dict(park_serviceArea_level_fre,orient='index')
    return park_serviceArea_level,park_serviceArea_level_fre_df

def multiple_line_graph_merged(df):
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np    
    
    # columnsName_sorted=list(df.columns)
    # columnsName_sorted.sort()
    # print(columnsName_sorted)    
    df=df.reindex(sorted(df.columns), axis=1)
    columnsName_sorted=list(df.columns)
    
    fig, ax=plt.subplots(1, 1, figsize=(30, 30))
    cmap=matplotlib.cm.get_cmap('tab20b') #'tab20b'
    colors=[cmap(i) for i in np.linspace(0,1,len(columnsName_sorted))] #These are the colors that will be used in the plot
    [i.set_visible(False) for i in ax.spines.values()] #Remove the plot frame lines. They are unnecessary here.
    ax.xaxis.tick_bottom() #Ensure that the axis ticks only show up on the bottom and left of the plot.
    ax.yaxis.tick_left()    
    fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94) #Limit the range of the plot to only where the data is.
    
    #Set the x/y-axis view limits.
    # ax.set_xlim(1969.5, 2011.1)
    # ax.set_ylim(-0.25, 90)    
    
    # Set a fixed location and format for ticks.
    ax.set_xticks(columnsName_sorted)
    y_val=range(df.min().min(), df.max().max(), 200)
    ax.set_yticks(y_val)    

    # Use automatic StrMethodFormatter creation
    # ax.xaxis.set_major_formatter('{x:.0f}')
    # ax.yaxis.set_major_formatter('{x:.0f}%')    
    
    ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3) #Provide tick lines across the plot to help your viewers trace along the axis ticks. Make sure that the lines are light and small so they don't obscure the primary data lines.

    #Remove the tick marks; they are unnecessary with the tick lines we just plotted. Make sure your axis ticks are large enough to be easily read. You don't want your viewers squinting to read your plot.
    ax.tick_params(axis='both', which='both', labelsize=14,bottom=False, top=False, labelbottom=True,left=False, right=False, labelleft=True)

    # df['park']=df.apply(lambda row:row.name.split("_")[-1],axis=1)
    majors=[n.split("_")[-1] for n in list(df.index)]
    print(majors)
    y_offsets={'Lotus Lake':0.5}
    
    for idx,val in df.iterrows():
        # print(idx,val)
        line,=ax.plot(columnsName_sorted,val,lw=2.5)
        
        y_pos = list(val)[-1] - 0.5
        ax.text(columnsName_sorted[-1]+1, y_pos, idx, fontsize=14, color=line.get_color())
        # break
    # fig.suptitle(" ", fontsize=18, ha="center")
    plt.show()    
 
def trim_axs(axs, N):
   """
   Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
   """
   axs = axs.flat
   for ax in axs[N:]:
       ax.remove()
   return axs[:N]      

def multiple_line_graph_separated(df):
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np    
    import math,copy
    
    # columnsName_sorted=list(df.columns)
    # columnsName_sorted.sort()
    # print(columnsName_sorted)    
    df=df.reindex(sorted(df.columns), axis=1)
    columnsName_sorted=list(df.columns)
    
    col_num=5
    fig, axs=plt.subplots(math.ceil(df.shape[0]/col_num), col_num, figsize=(30, 25),constrained_layout=True)
    axs=trim_axs(axs, df.shape[0])
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    cmap=matplotlib.cm.get_cmap('gist_ncar') #'tab20b'
    colors=[cmap(i) for i in np.linspace(0,1,df.shape[0])] #These are the colors that will be used in the plot
    
    # for sub_idx in range(df.shape[0]) :
    sub_idx=0    
    for idx,val in df.iterrows():
        ax=axs[sub_idx]
    
        [i.set_visible(False) for i in ax.spines.values()] #Remove the plot frame lines. They are unnecessary here.
        ax.xaxis.tick_bottom() #Ensure that the axis ticks only show up on the bottom and left of the plot.
        ax.yaxis.tick_left()    
        fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94) #Limit the range of the plot to only where the data is.
        
        #Set the x/y-axis view limits.
        # ax.set_xlim(1969.5, 2011.1)
        # ax.set_ylim(-0.25, 90)    
        
        # Set a fixed location and format for ticks.
        ax.set_xticks(columnsName_sorted[::2])
        y_val=range(df.min().min(), df.max().max(), 500)
        ax.set_yticks(y_val)    
    
        # Use automatic StrMethodFormatter creation
        # ax.xaxis.set_major_formatter('{x:.0f}')
        # ax.yaxis.set_major_formatter('{x:.0f}%')    
        
        ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3) #Provide tick lines across the plot to help your viewers trace along the axis ticks. Make sure that the lines are light and small so they don't obscure the primary data lines.
    
        #Remove the tick marks; they are unnecessary with the tick lines we just plotted. Make sure your axis ticks are large enough to be easily read. You don't want your viewers squinting to read your plot.
        ax.tick_params(axis='both', which='both', labelsize=14,bottom=False, top=False, labelbottom=True,left=False, right=False, labelleft=True)
    
        # df['park']=df.apply(lambda row:row.name.split("_")[-1],axis=1)        
        y_offsets={'Lotus Lake':0.5}
        
        # print(idx,val)
        line,=ax.plot(columnsName_sorted,val,lw=2.5,color='k')   #colors[sub_idx]      
        val_pos=copy.deepcopy(list(val))
        val_pos.sort()        
        # print(idx)
        y_pos = val_pos[-2] - 0.5
        if idx in ['22_Liuhe Phoenix Mountain','47_South Park of Sports Institute','50_Chrysanthemums Terrace']:
            x_pos=columnsName_sorted[-13]-1
        else:
            x_pos=columnsName_sorted[-8]+1
        ax.text(x_pos, y_pos, idx.split("_")[-1], fontsize=14, color=line.get_color())
        sub_idx+=1
        # break
        # fig.suptitle(" ", fontsize=18, ha="center")
    plt.savefig('./graph/park_serviceArea_level_fre.png') 
    plt.show()            

if __name__=="__main__":
    # comprehensivePark_adjacentStations=postSQL2gpd(table_name='adjacent_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    # start_stops_PointUid=flatten_lst([eval(lst) for lst in comprehensivePark_adjacentStations.adjacent_PointUid])

    #NETWORK ANALYSIS
    # G_SB=nx.read_gpickle("./network/G_SB.gpickle")    
    # bus_shortest_paths_loop(comprehensivePark_adjacentStations,G_SB,save_root='processed data/shortest_path')

    # with open(os.path.join('processed data/shortest_path','fn_lst.pkl'),'rb') as f:
    #     shortestPaths_fn_lst=pickle.load(f)
    # park_station_shortestPaths(shortestPaths_fn_lst,nanjing_epsg)
    # gpd2postSQL(a,table_name='temp',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')    
    
    # comprehensive_park_en=postSQL2gpd(table_name='comprehensive_park_en',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    # with open('./processed data/park_shortest_path.pkl','rb') as f:
    #     park_shortest_path=pickle.load(f)
        
    # stats,comprehensive_park_en_timecost=stats_shortest_length(park_shortest_path,comprehensive_park_en)
    # gpd2postSQL(comprehensive_park_en_timecost,table_name='comprehensive_park_en',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')

    #service area levels
    # park_serviceArea_level,park_serviceArea_level_fre=service_area_levels(G_SB,park_shortest_path,comprehensivePark_adjacentStations,SES=[0,90,5],figsize=(30*2, 30*2),level=3)
    # with open('./processed data/park_serviceArea_level.pkl','wb') as f:
    #     pickle.dump(park_serviceArea_level,f)    
    # with open('./processed data/park_serviceArea_level_fre.pkl','wb') as f:
    #     pickle.dump(park_serviceArea_level_fre,f)      
        
    # with open('./processed data/park_serviceArea_level.pkl','rb') as f:
    #     park_serviceArea_level=pickle.load(f)    
    with open('./processed data/park_serviceArea_level_fre.pkl','rb') as f:
        park_serviceArea_level_fre=pickle.load(f)   
    
    multiple_line_graph_separated(copy.deepcopy(park_serviceArea_level_fre))