# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:08:28 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from database import postSQL2gpd,gpd2postSQL
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def idx_clustering(idxes_df,field,n_clusters=10):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj
    
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # print(pts_geometry)
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    # print(pts_coordis)
    
    nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
    connectivity=nbrs.kneighbors_graph(pts_coordis)
    # print(connectivity.toarray())
    
    X=np.expand_dims(idxes_df[field].to_numpy(),axis=1)
    # print(X.shape)
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    # print(clustering.labels_.shape)
    idxes_df['clustering_'+field]=clustering.labels_
    
    mean=idxes_df.groupby(['clustering_'+field])[field].mean() #.reset_index()
    idxes_df['clustering_'+field+'_mean']=idxes_df['clustering_'+field].map(mean.to_dict())
    
    wgs84=pyproj.CRS('EPSG:4326')
    idxes_df_gdf=gpd.GeoDataFrame(idxes_df,geometry=idxes_df.geometry,crs=wgs84)     
    return idxes_df_gdf

def idxes_clustering(idxes_df,fields,n_clusters=10):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj
    from sklearn.preprocessing import normalize
    
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # print(pts_geometry)
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    # print(pts_coordis)
    
    nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
    connectivity=nbrs.kneighbors_graph(pts_coordis)
    # print(connectivity.toarray())
    
    X_=idxes_df[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')
    # print(X.shape)
    # print(idxes_df[fields].to_numpy().shape)
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    # print(clustering.labels_.shape)
    idxes_df['clustering']=clustering.labels_
    idxes_df['clustering_']=idxes_df.clustering.apply(lambda row:row+1)
    
    wgs84=pyproj.CRS('EPSG:4326')
    xian_epsg=pyproj.CRS('EPSG:32649') #Xi'an   WGS84 / UTM zone 49N    
    idxes_df_gdf=gpd.GeoDataFrame(idxes_df,geometry=idxes_df.geometry,crs=wgs84)   
    idxes_df_gdf=idxes_df_gdf.to_crs(xian_epsg)
    
    return idxes_df_gdf

def Nclusters_sihouette_analysis(idxes_df,fields,range_n_clusters=[2,3,4,5,6]):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    
    for n_clusters in tqdm(range_n_clusters):    
        pts_geometry=idxes_df[['geometry']]    
        pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
        # print(pts_geometry)
        pts_coordis=pts_geometry[['x','y']].to_numpy()
        # print(pts_coordis)
        
        nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
        connectivity=nbrs.kneighbors_graph(pts_coordis)
        # print(connectivity.toarray())
        
        X_=idxes_df[fields].to_numpy()
        X=normalize(X_,axis=0, norm='max')
        # X=PCA(n_components=2).fit_transform(X_)
        # print(X.shape)
        # print(idxes_df[fields].to_numpy().shape)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])        
        
        clusterer=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters,compute_distances=True).fit(X)
        print('distance:{}'.format(clusterer.distances_))
        # clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10)
        
        cluster_labels=clusterer.fit_predict(X)
        
        
        silhouette_avg=silhouette_score(X, cluster_labels)  # The silhouette_score gives the average value for all the samples.
        print('\n')
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg) # Compute the silhouette scores for each sample        
        sample_silhouette_values=silhouette_samples(X, cluster_labels)
        
        y_lower=10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples        
            
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # # Draw white circles at cluster centers
        # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
        #             c="white", alpha=1, s=200, edgecolor='k')
    
        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
        #                 s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold') 
        # plt.show()
        
        
        # Z = linkage(X, method='ward')
        # plt.figure()
        # dendrogram(Z)
        # plt.show()
        
        # break  
    plt.show()
    
def idxes_clustering_contribution(idxes_df,fields,n_clusters=10):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    from sklearn import cluster
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj
    from sklearn.feature_selection import chi2, SelectKBest, f_classif
    from sklearn import preprocessing
    from sklearn.preprocessing import normalize
    
    from yellowbrick.cluster import KElbowVisualizer    
    from yellowbrick.features import Manifold


    # import matplotlib    
    # font = {
    #         # 'family' : 'normal',
    #         # 'weight' : 'bold',
    #         'size'   : 28}
    # matplotlib.rc('font', **font) 
    
    pts_geometry=idxes_df[['geometry']]    
    pts_geometry[['x','y']]=pts_geometry.geometry.apply(lambda row:pd.Series([row.x,row.y]))
    # print(pts_geometry)
    pts_coordis=pts_geometry[['x','y']].to_numpy()
    # print(pts_coordis)
    
    nbrs=NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(pts_coordis)
    connectivity=nbrs.kneighbors_graph(pts_coordis)
    # print(connectivity.toarray())
    
    
    X_=idxes_df[fields].to_numpy()
    X=normalize(X_,axis=0, norm='max')
    
    # print(X.shape)
    # print(idxes_df[fields].to_numpy().shape)
    clustering=cluster.AgglomerativeClustering(connectivity=connectivity,n_clusters=n_clusters).fit(X)
    # print(clustering.labels_.shape)
    # idxes_df['clustering']=clustering.labels_
    
    # wgs84=pyproj.CRS('EPSG:4326')
    # idxes_df_gdf=gpd.GeoDataFrame(idxes_df,geometry=idxes_df.geometry,crs=wgs84)   
    
    y=clustering.labels_
    selector=SelectKBest(score_func=f_classif, k=len(fields)) #score_func=chi2    
    selector.fit(X,y)
    # scores=-np.log10(selector.pvalues_)
    # scores /= scores.max()
    
    # X_indices = np.arange(X.shape[-1])
    # print(scores)
    # plt.bar(X_indices - .45, scores, width=.2,label=r'Univariate score ($-Log(p_{value})$)')    
    
    dfscores = pd.DataFrame(selector.scores_)
    dfpvalues=pd.DataFrame(selector.pvalues_)
    dfcolumns = pd.DataFrame(fields)  
    featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
    featureScores.columns = ['Factor','Score','p_value']  #naming the dataframe columns
    print(featureScores)
    # featureScores.to_excel('./graph/tl_features scores.xlsx') #'./graph/features scores.xlsx'
    
    # print(featureScores.nlargest(10,'Score'))  #print 10 best features
    
    featureScores_=featureScores.set_index('Factor')    
    featureScores_.nlargest(len(fields),'Score').Score.plot(kind='barh',figsize=(30,20),fontsize=38)
    featureScores_.Score.plot(kind='barh')
    plt.show()    
    
    clustering_=cluster.AgglomerativeClustering(connectivity=connectivity,) #n_clusters=n_clusters
    visualizer = KElbowVisualizer(clustering_, timings=False,size=(500, 500), ) #k=(4,12) metric='calinski_harabasz'
    visualizer.fit(X)    # Fit the data to the visualizer
    visualizer.show(outpath="./graph/tl_KEIbow_.png")    # Finalize and render the figure
    
    # Instantiate the visualizer
    # viz = Manifold(manifold="isomap", n_neighbors=10)
    # viz.fit_transform(X, y)  # Fit the data to the visualizer
    # viz.show()               # Finalize and render the figure    
    
    
    

if __name__=="__main__":
    '''
    panorama_object_percent_gdf=postSQL2gpd(table_name='panorama_object_percent',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    sky_class_level_metrics_gdf=postSQL2gpd(table_name='sky_class_level_metrics',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    colors_dominant_entropy_gdf=postSQL2gpd(table_name='colors_dominant_entropy',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    kp_size_stats_gdf=postSQL2gpd(table_name='kp_size_stats',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    
    idx_panorama_object_percent=['vege','sky','ground','equilibrium_degree']
    idx_sky_class_level_metrics=['number_of_patches','perimeter_area_ratio_mn','shape_index_mn','fractal_dimension_mn']
    idx_colors_dominant_entropy=['counter']
    idx_kp_size_stats=['-0.001_10.0', '10.0_20.0','30.0_40.0', '20.0_30.0']
    
    idx_auxiliary=['fn_stem', 'fn_key', 'fn_idx', 'geometry']

    idxes_merge=pd.concat([panorama_object_percent_gdf[idx_panorama_object_percent],
                            sky_class_level_metrics_gdf[idx_sky_class_level_metrics],
                            colors_dominant_entropy_gdf[idx_colors_dominant_entropy],
                            kp_size_stats_gdf[idx_kp_size_stats],
                            panorama_object_percent_gdf[idx_auxiliary]],axis=1)
    fields_mapping={'vege':'Green view index', 'sky':'Sky view factor', 'ground':'Ground view index', 'equilibrium_degree':'Equilibrium degree',
                   'number_of_patches':'number of patches', 'perimeter_area_ratio_mn':'Perimeter area ratio(mn)', 'shape_index_mn':'Shape index(mn)',
                   'fractal_dimension_mn':'Fractal dimension(mn)', 'counter':'Color richness index', 
                   '-0.001_10.0':'Key point size(0-10]', '10.0_20.0':'Key point size(10-20]','30.0_40.0':'Key point size(30-40]', '20.0_30.0':'Key point size(20-30]'}  
    idxes_merge=idxes_merge.rename(columns=fields_mapping)
    
    # idx_clustering_gdf=idx_clustering(idxes_merge.copy(deep=True),field='Green view index',n_clusters=8)
    
    fields=['Green view index', 'Sky view factor', 'Ground view index',
       'Equilibrium degree', 'number of patches', 'Perimeter area ratio(mn)',
       'Shape index(mn)', 'Fractal dimension(mn)', 'Color richness index',
       'Key point size(0-10]', 'Key point size(10-20]',
       'Key point size(30-40]', 'Key point size(20-30]',]
    n_clusters=8
    idxes_clustering_gdf=idxes_clustering(idxes_merge.copy(deep=True),fields,n_clusters=n_clusters)
    gpd2postSQL(idxes_clustering_gdf,table_name='idxes_clustering_{}'.format(n_clusters),myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # plt.figure(figsize=(20,20))
    # sns.set(font_scale=1.8)
    # sns_plot=sns.heatmap(idxes_merge[fields].corr(),annot=True,cmap="RdYlGn")
    # plt.savefig('./graph/idxes_corri.png',dpi=300)
    
    # Nclusters_sihouette_analysis(idxes_merge.copy(deep=True),fields=['Sky view factor','Key point size(0-10]'],range_n_clusters=[2,3,4,5,6,7,8,9,10])   
    
    idxes_clustering_contribution(idxes_merge.copy(deep=True),fields,n_clusters=8)
    '''
    
    
    #street scale
    panorama_object_percent_gdf=postSQL2gpd(table_name='tourLine_panorama_object_percent',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    sky_class_level_metrics_gdf=postSQL2gpd(table_name='tl_sky_class_level_metrics',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    colors_dominant_entropy_gdf=postSQL2gpd(table_name='tl_colors_dominant_entropy',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    kp_size_stats_gdf=postSQL2gpd(table_name='tl_kp_size_stats',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    
    idx_panorama_object_percent=['vege','sky','ground','equilibrium_degree']
    idx_sky_class_level_metrics=['number_of_patches','perimeter_area_ratio_mn','shape_index_mn','fractal_dimension_mn']
    idx_colors_dominant_entropy=['counter']
    idx_kp_size_stats=['-0.001_10.0', '10.0_20.0','30.0_40.0', '20.0_30.0']
    
    idx_auxiliary=['fn_stem', 'fn_key', 'fn_idx', 'geometry']

    idxes_merge=pd.concat([panorama_object_percent_gdf[idx_panorama_object_percent],
                            sky_class_level_metrics_gdf[idx_sky_class_level_metrics],
                            colors_dominant_entropy_gdf[idx_colors_dominant_entropy],
                            kp_size_stats_gdf[idx_kp_size_stats],
                            panorama_object_percent_gdf[idx_auxiliary]],axis=1)
    fields_mapping={'vege':'Green view index', 'sky':'Sky view factor', 'ground':'Ground view index', 'equilibrium_degree':'Equilibrium degree',
                   'number_of_patches':'number of patches', 'perimeter_area_ratio_mn':'Perimeter area ratio(mn)', 'shape_index_mn':'Shape index(mn)',
                   'fractal_dimension_mn':'Fractal dimension(mn)', 'counter':'Color richness index', 
                   '-0.001_10.0':'Key point size(0-10]', '10.0_20.0':'Key point size(10-20]','30.0_40.0':'Key point size(30-40]', '20.0_30.0':'Key point size(20-30]'}  
    idxes_merge=idxes_merge.rename(columns=fields_mapping)
    
    # idx_clustering_gdf=idx_clustering(idxes_merge.copy(deep=True),field='Green view index',n_clusters=8)
    
    fields=['Green view index', 'Sky view factor', 'Ground view index',
       'Equilibrium degree', 'number of patches', 'Perimeter area ratio(mn)',
       'Shape index(mn)', 'Fractal dimension(mn)', 'Color richness index',
       'Key point size(0-10]', 'Key point size(10-20]',
       'Key point size(30-40]', 'Key point size(20-30]',]
    n_clusters=5
    # idxes_clustering_gdf=idxes_clustering(idxes_merge.copy(deep=True),fields,n_clusters=n_clusters)
    # gpd2postSQL(idxes_clustering_gdf,table_name='tl_idxes_clustering_{}'.format(n_clusters),myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    # plt.figure(figsize=(20,20))
    # sns.set(font_scale=1.8)
    # sns_plot=sns.heatmap(idxes_merge[fields].corr(),annot=True,cmap="RdYlGn")
    # plt.savefig('./graph/idxes_corri.png',dpi=300)
    
    # Nclusters_sihouette_analysis(idxes_merge.copy(deep=True),fields=['Sky view factor','Key point size(0-10]'],range_n_clusters=[2,3,4,5,6,7,8,9,10])   
    
    idxes_clustering_contribution(idxes_merge.copy(deep=True),fields,n_clusters=5)
    