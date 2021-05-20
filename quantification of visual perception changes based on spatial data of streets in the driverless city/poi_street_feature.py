# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:28:35 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import pickle
from database import postSQL2gpd,gpd2postSQL
import pandas as pd

xian_epsg=32649 #Xi'an   WGS84 / UTM zone 49N
wgs84_epsg=4326

poi_classificationName={
        0:"delicacy",
        1:"hotel",
        2:"shopping",
        3:"lifeService",
        4:"beauty",
        5:"spot",
        6:"entertainment",
        7:"sports",
        8:"education",
        9:"media",
        10:"medicalTreatment",
        11:"carService",
        12:"trafficFacilities",
        13:"finance",
        14:"realEstate",
        15:"corporation",
        16:"government",
        17:"entrance",
        18:"naturalFeatures",        
        }
poi_classificationName_reverse={v:k for k,v in poi_classificationName.items()}

def street_poi_structure(poi,position,distance=300):
    from tqdm import tqdm
    import pickle,math
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    # tqdm.pandas()
    poi_num=len(poi_classificationName.keys())    
    feature_vector=np.zeros(poi_num)
    
    poi_=poi.copy(deep=True)
    pos_poi_dict={}
    pos_poi_idxes_df=pd.DataFrame(columns=['geometry','frank_e','num'])
    pos_poi_feature_vector_df=pd.DataFrame(columns=['geometry']+list(range(poi_num)))
    # print(pos_poi_feature_vector)
    for idx,row in tqdm(position.iterrows(),total=position.shape[0]):
        poi_['within']=poi_.geometry.apply(lambda pt: pt.within(row.geometry.buffer(distance)))
        # print(poi_)
        poi_selection_df=poi_[poi_['within']==True]
        counts=poi_selection_df.level_0.value_counts().to_dict()
        num=len(poi_selection_df)
        counts_percent={k:v/num for k,v in counts.items()}
        # print(counts_percent)
        
        ve=0.0
        for v in counts_percent.values():
            if v!=0.:
                ve-=v*math.log(v)
        max_entropy=math.log(num)
        frank_e=ve/max_entropy*100        
        # print(max_entropy,frank_e)
        
        for k,v in counts.items(): #计算特征聚类出现的频数/直方图
            poi_name=k.split("_")[-1]
            poi_idx=poi_classificationName_reverse[poi_name]
            # print(poi_idx,v)
            feature_vector[poi_idx]=v        
        # print(feature_vector)
        pos_poi_dict.update({idx:{'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx ,'counts':counts,'counts_percent':counts_percent,'feature_vector':feature_vector,'num':num,'frank_e':frank_e,'geometry':row.geometry}})
        pos_poi_idxes_df=pos_poi_idxes_df.append({'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx,'geometry':row.geometry,'frank_e':frank_e,'num':num},ignore_index=True)
        feature_vector_dict={i:feature_vector[i] for i in range(len(feature_vector))}
        feature_vector_dict.update({'geometry':row.geometry,'fn_stem':row.fn_stem, 'fn_key':row.fn_key, 'fn_idx':row.fn_idx,})
        pos_poi_feature_vector_df=pos_poi_feature_vector_df.append(feature_vector_dict,ignore_index=True)
        
        # if idx==3:break        
    pos_poi_idxes_gdf=gpd.GeoDataFrame(pos_poi_idxes_df,geometry=pos_poi_idxes_df.geometry,crs=position.crs)   
    pos_poi_idxes_gdf['num_diff']=pos_poi_idxes_gdf.num.diff()
    pos_poi_feature_vector_gdf=gpd.GeoDataFrame(pos_poi_feature_vector_df,geometry=pos_poi_feature_vector_df.geometry,crs=position.crs)   
    with open('./processed data/pos_poi_dict.pkl','wb') as f:
        pickle.dump(pos_poi_dict,f)    
        
    return pos_poi_idxes_gdf,pos_poi_feature_vector_gdf


def poi_feature_clustering(feature_vector,fields,n_clusters=7,feature_analysis=True):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj
    from yellowbrick.cluster import KElbowVisualizer    
    from yellowbrick.features import Manifold
    from sklearn.feature_selection import chi2, SelectKBest, f_classif
    from sklearn import preprocessing
    from sklearn.preprocessing import normalize
    import matplotlib.pyplot as plt
    
    pts_geometry=feature_vector[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # print(pts_geometry)
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    # print(pts_coordis)
    
    nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
    connectivity=nbrs.kneighbors_graph(pts_coordis)
    # print(connectivity.toarray())    
    X_=feature_vector[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')    

    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    feature_vector['clustering']=clustering.labels_
    
    #_________________________________________________________________________
    if feature_analysis==True:
        y=clustering.labels_
        selector=SelectKBest(score_func=f_classif, k=len(fields)) #score_func=chi2    
        selector.fit(X,y)
        
        dfscores = pd.DataFrame(selector.scores_)
        dfpvalues=pd.DataFrame(selector.pvalues_)
        dfcolumns = pd.DataFrame(fields)  
        featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
        featureScores.columns = ['Factor','Score','p_value']  #naming the dataframe columns
        featureScores['Factor']=featureScores['Factor'].apply(lambda row:int(row))
        featureScores['poi_name']=featureScores['Factor'].map(poi_classificationName)
        featureScores=featureScores.sort_values(by=['Score'])
        # print(type(featureScores['Factor'][0]))
        print(featureScores)
        # featureScores.to_excel('./graph/tl_poi_features scores.xlsx') 
        
        featureScores_=featureScores.set_index('Factor')    
        featureScores_.nlargest(len(fields),'Score').Score.plot(kind='barh',figsize=(30,20),fontsize=38)
        featureScores_.Score.plot(kind='barh')
        plt.show()    
        
        clustering_=cluster.AgglomerativeClustering(connectivity=connectivity,) #n_clusters=n_clusters
        visualizer = KElbowVisualizer(clustering_, timings=False,size=(500, 500), k=(4,12)) #k=(4,12) metric='calinski_harabasz'
        visualizer.fit(X)    # Fit the data to the visualizer
        # visualizer.show(outpath="./graph/tl_poi_clustering_KEIbow_.png")    # Finalize and render the figure    
        
    return feature_vector     
    

if __name__=="__main__":
    # poi_gdf=postSQL2gpd(table_name='poi',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # poi_gdf=poi_gdf.to_crs(xian_epsg)
    # tl_idxes_clustering_12_gdf=postSQL2gpd(table_name='tl_idxes_clustering_12',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # pos_poi_idxes_gdf,pos_poi_feature_vector_gdf=street_poi_structure(poi=poi_gdf,position=tl_idxes_clustering_12_gdf)
    # gpd2postSQL(pos_poi_idxes_gdf,table_name='pos_poi_idxes',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # gpd2postSQL(pos_poi_feature_vector_gdf,table_name='pos_poi_feature_vector',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # with open('./processed data/pos_poi_dict.pkl','rb') as f:
    #     pos_poi_dict=pickle.load(f)
    
    pos_poi_feature_vector_gdf=postSQL2gpd(table_name='pos_poi_feature_vector',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    fields=[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16', '17', '18']
    n_clusters=12  #12
    feature_vector=poi_feature_clustering(pos_poi_feature_vector_gdf,fields,n_clusters=n_clusters,feature_analysis=True)
    # gpd2postSQL(feature_vector,table_name='pos_poi_feature_vector_{}'.format(n_clusters),myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

