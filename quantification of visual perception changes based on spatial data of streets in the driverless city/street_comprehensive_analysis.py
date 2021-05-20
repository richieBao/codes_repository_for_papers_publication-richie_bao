# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:39:49 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from database import postSQL2gpd,gpd2postSQL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

poi_classificationName={
        0:"delicacy",
        1:"hotel",
        2:"shopping",
        3:"life service",
        4:"beauty",
        5:"spot",
        6:"entertainment",
        7:"sports",
        8:"education",
        9:"media",
        10:"medical treatment",
        11:"car service",
        12:"traffic facilities",
        13:"finance",
        14:"real estate",
        15:"corporation",
        16:"government",
        17:"entrance",
        18:"natural features",        
        }

def clustering_POI_stats(df,num=3):
    from matplotlib import cm
    
    stats_sum=df.groupby(['clustering_POI']).sum()
    stats_sum=stats_sum.rename(columns={str(k):v for k,v in poi_classificationName.items()})
    # stats_sum=stats_sum.T
    print(stats_sum)
    
    # plot=stats_sum.plot(kind='pie',subplots=True, figsize=(60, 20),legend=False)
    fig, ax=plt.subplots(figsize=(30, 12),) #figsize=(40, 20),
    cmap=cm.get_cmap('tab20') # Colour map (there are many others)
    plot=stats_sum.plot(kind='bar',stacked=True, legend=True,ax=ax,rot=0,fontsize=35,cmap=cmap)
    ax.set_facecolor("w")
    plt.legend( prop={"size":20})
    plt.savefig('./graph/clustering_POI_stats_7.png',dpi=300)
    
    # POI_clustering_stats={}
    # for idx,row in stats_sum.iterrows():
    #     idxes=row.nlargest(num).index.to_list()
    #     name=[poi_classificationName[int(i)] for i in idxes]
    #     # print(name)
    #     POI_clustering_stats[idx]=name
        
    #     # break
    # print(POI_clustering_stats)



if __name__=="__main__":
    tl_idxes_clustering_gdf=postSQL2gpd(table_name='tl_idxes_clustering_12',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    tl_idxes_clustering_gdf=tl_idxes_clustering_gdf.rename(columns={'clustering':'clustering_idxes'})
    selection_tl_idxes_clustering=['Green view index', 'Sky view factor', 'Ground view index',
       'Equilibrium degree', 'number of patches', 'Perimeter area ratio(mn)','Shape index(mn)', 'Fractal dimension(mn)', 'Color richness index',
       'Key point size(0-10]', 'Key point size(10-20]','Key point size(30-40]', 'Key point size(20-30]', 'clustering_idxes']
    
    #_12
    pos_poi_feature_vector_gdf=postSQL2gpd(table_name='pos_poi_feature_vector',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    pos_poi_feature_vector_gdf=pos_poi_feature_vector_gdf.rename(columns={'clustering':'clustering_POI'})
    selection_pos_poi_feature_vector=['clustering_POI'] #'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16', '17', '18',
    
    pos_poi_idxes_gdf=postSQL2gpd(table_name='pos_poi_idxes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    pos_poi_idxes_gdf=pos_poi_idxes_gdf.rename(columns={'frank_e':'frank e_POI','num':'num_POI'})
    selection_pos_poi_idxes=['frank e_POI', 'num_POI']

    street_idxes=pd.concat([tl_idxes_clustering_gdf[selection_tl_idxes_clustering],
                            pos_poi_feature_vector_gdf[selection_pos_poi_feature_vector],
                            pos_poi_idxes_gdf[selection_pos_poi_idxes]],axis=1)

    # plt.figure(figsize=(20,20))
    # sns.set(font_scale=1.8)
    # sns_plot=sns.heatmap(street_idxes.corr(),annot=True,cmap="RdYlGn")
    # plt.savefig('./graph/idxes_corri.png',dpi=300)
    
    selection_stats=['clustering_POI','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16', '17', '18']
    clustering_POI_stats(pos_poi_feature_vector_gdf[selection_stats])
    
    
    