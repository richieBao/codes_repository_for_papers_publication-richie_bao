# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:49:39 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""

from database import postSQL2gpd
from scipy import stats


def df_print(df,columns,sort_by,x_idx,new_legend,figsize,normalize=True):
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np   

    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.facecolor']='white'
    
    if len(x_idx)==1:
        df.set_index(x_idx[0],inplace=True)
    else:
        df['x_idx']=df.apply(lambda row:':'.join([str(s) for s in row[x_idx].to_list()]),axis=1)
        df.set_index(df['x_idx'],inplace=True)
        
    df_plot=df[columns].sort_values(by=sort_by)
    # print(df_plot.index)
    if normalize:       
        columns_norm=[column+'_norm' for column in df_plot]
        min_max_scaler=preprocessing.MinMaxScaler()
        norm_v_list=[]
        for c in columns:
            norm_values=min_max_scaler.fit_transform(df_plot[c].values.reshape(-1,1)).reshape(-1)
            norm_v_list.append(norm_values)
        df_norm=pd.DataFrame(np.stack(norm_v_list,axis=-1),columns=columns_norm,index=df_plot.index)   
        # print(df_norm)            
        ax=df_norm.plot(marker='o',figsize=figsize)
        # print(len(df_norm.index))
        ax.set_xticks(list(range(len(df_norm.index))))
        ax.set_xticklabels(df_norm.index,rotation=90)
        ax.xaxis.label.set_visible(False)   
        
        ax.patch.set_facecolor('white')
        
        if new_legend:
            ax.legend(new_legend,loc='upper left')        
    
    else:
        ax=df_plot.plot()
        ax.set_xticks(list(range(len(df_plot.index))))
        ax.set_xticklabels(df_plot.index,rotation=90)
        ax.xaxis.label.set_visible(False)



if __name__=="__main__":
    adjacent_stations=postSQL2gpd(table_name='adjacent_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='public_transport_accessibility')
    df_print(adjacent_stations,['adjacent_perimeterRatio','adjacent_areaRatio',],sort_by=['adjacent_perimeterRatio'],x_idx=['Name_EN','adjacent_num'],
             new_legend=['Number of stations in the vicinity/park perimeter','Number of stations in the vicinity/park area',],figsize=(15,15),normalize=True) #'adjacent_areaRatio',
    #'adjacent_num';'Normalization of the parks number'
    
    print(stats.pearsonr(adjacent_stations['adjacent_num'],adjacent_stations['park_perimeter']))
    print(stats.pearsonr(adjacent_stations['adjacent_num'],adjacent_stations['park_area']))
    print(stats.pearsonr(adjacent_stations['park_perimeter'],adjacent_stations['park_area']))
