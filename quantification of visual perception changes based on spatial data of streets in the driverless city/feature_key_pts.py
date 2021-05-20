# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:50:24 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import glob, os
import pickle
from database import postSQL2gpd,gpd2postSQL
import numpy as np

class feature_builder_BOW:
    '''
    class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
    '''   
    def __init__(self,num_cluster=32):
        self.num_clusters=num_cluster

    def extract_features(self,img):
        import cv2 as cv
        '''
        function - 提取图像特征
        
        Paras:
        img - 读取的图像
        '''
        #A
        # star_detector=cv.xfeatures2d.StarDetector_create()
        # key_points=star_detector.detect(img)
        # img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征 

        #B 
        # Initiate FAST detector
        star=cv.xfeatures2d.StarDetector_create() 
        # find the keypoints with STAR
        key_point=star.detect(img)
        cv.drawKeypoints(img,key_point,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Initiate BRIEF extractor
        brief=cv.xfeatures2d.BriefDescriptorExtractor_create()    
        # compute the descriptors with BRIEF
        kp, des=brief.compute(img, key_point)   

        return des,kp
    
    def visual_BOW(self,des_all):
        from sklearn.cluster import KMeans
        '''
        function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋
        
        des_all - 所有图像的关键点描述子
        '''
        print("start KMean...")
        kmeans=KMeans(self.num_clusters)
        kmeans=kmeans.fit(des_all)
        #centroids=kmeans.cluster_centers_
        print("end KMean...")
        return kmeans         
    
    def get_visual_BOW(self,training_data):
        import cv2 as cv
        from tqdm import tqdm
        '''
        function - 提取图像特征，返回所有图像关键点聚类视觉词袋
        
        Paras:
        training_data - 训练数据集
        '''
        des_all=[]
        #i=0        
        for item in tqdm(training_data):
            img=cv.imread(item)
            img=img[:int(img.shape[0]*(70/100))]  
            
            des,_=self.extract_features(img)
            des_all.extend(des)           
            #print(des.shape)

            #if i==10:break
            #i+=1        
        kmeans=self.visual_BOW(des_all)      
        return kmeans
    
    def normalize(self,input_data):
        import numpy as np
        '''
        fuction - 归一化数据
        
        input_data - 待归一化的数组
        '''
        sum_input=np.sum(input_data)
        if sum_input>0:
            return input_data/sum_input #单一数值/总体数值之和，最终数值范围[0,1]
        else:
            return input_data               
    
    def construct_feature(self,img,kmeans):
        import numpy as np
        '''
        function - 使用聚类的视觉词袋构建图像特征（构造码本）
        
        Paras:
        img - 读取的单张图像
        kmeans - 已训练的聚类模型
        '''
        des,kp=self.extract_features(img)
        labels=kmeans.predict(des.astype(np.float)) #对特征执行聚类预测类标
        # print(len(labels))
        feature_vector=np.zeros(self.num_clusters)
        for i,item in enumerate(labels): #计算特征聚类出现的频数/直方图
            # print(labels[i])
            feature_vector[labels[i]]+=1
        feature_vector_=np.reshape(feature_vector,((1,feature_vector.shape[0])))
        # return self.normalize(feature_vector_)
        return feature_vector_,labels,kp
    
    def get_feature_map(self,training_data,kmeans):
        import cv2 as cv
        from pathlib import Path
        from tqdm import tqdm
        '''
        function - 返回每个图像的特征映射（码本映射）
        Paras:
        training_data - 训练数据集
        kmeans - 已训练的聚类模型
        '''
        feature_map=[]
        for item in tqdm(training_data):            
            fn_stem=Path(item).stem
            fn_key,fn_idx=fn_stem.split("_")
            
            temp_dict={}
            temp_dict['fn_stem']=fn_stem
            #print("Extracting feature for",item['image_path'])
            img=cv.imread(item)
            img=img[:int(img.shape[0]*(70/100))]  
            
            feature_vector,labels,kp=self.construct_feature(img,kmeans)
            temp_dict['feature_vector']=feature_vector
            temp_dict['labels']=labels
            # print(kp)
            temp_dict['kp']=[{'pt':kp[i].pt, 
                             'size':kp[i].size, 
                             'angle':kp[i].angle,
                             'response':kp[i].response, 
                             'octave':kp[i].octave, 
                             'class_id':kp[i].class_id} for i in range(len(kp))]
            if temp_dict['feature_vector'] is not None:
                feature_map.append(temp_dict)
        #print(feature_map[0]['feature_vector'].shape,feature_map[0])
        return feature_map

def STAR_detection(img_fp,save=False):
    import cv2 as cv
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    '''
    function - 使用Star特征检测器提取图像特征
    '''
    img=cv.imread(img_fp)
    img=img[:int(img.shape[0]*(70/100))]   
    
    
    # Initiate FAST detector
    star=cv.xfeatures2d.StarDetector_create() 
    # find the keypoints with STAR
    key_points=star.detect(img)
    cv.drawKeypoints(img,key_points,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #A
    # Initiate BRIEF extractor
    brief=cv.xfeatures2d.BriefDescriptorExtractor_create()    
    # compute the descriptors with BRIEF
    kp, des=brief.compute(img, key_points)
    
    # #B    
    # img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征     


    if save:
        cv.imshow('star features',img_copy)
        cv.imwrite('./data/star_features.jpg',img) #保存图像
        cv.waitKey()
    else:        
        fig, ax=plt.subplots(figsize=(30,15))
        ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB) )
        plt.show()      
    
    print('key pts num={}'.format(len(key_points)))
    # print(help(key_point[0]))
    example=key_points[10]
    print('Data descriptors,angle={},class_id={}, octave={},pt={},response={},size={}'.format(example.angle,example.class_id,example.octave,example.pt,example.response,example.size))
    
    print(des.shape)
    return kp

def kps_desciptors_BOW_feature(feature_map,coords,num_cluster=32):
    from tqdm import tqdm
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj
    
    panorama_object_df=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(num_cluster)))
    for feature_info in tqdm(feature_map):
        fn_stem=feature_info['fn_stem']
        fn_key,fn_idx=fn_stem.split("_")
        # print(fn_stem)
        # print(feature_info['feature_vector'].tolist())
        featureMap_dict=dict(zip(list(range(num_cluster)),feature_info['feature_vector'].tolist()[0]))
        # print(featureMap_dict)
        coord=coords[fn_key][int(fn_idx)]
        featureMap_dict.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})   
        panorama_object_df=panorama_object_df.append(featureMap_dict,ignore_index=True)
        # break
    wgs84=pyproj.CRS('EPSG:4326')
    featureMap_gdf=gpd.GeoDataFrame(panorama_object_df,geometry=panorama_object_df.geometry,crs=wgs84)     
    
    return featureMap_gdf

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    import cv2
    
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kp_stats(feature_map,coords):    
    from tqdm import tqdm
    import pandas as pd 
    from shapely.geometry import Point
    import geopandas as gpd
    import pyproj    
    
    kp_dict={i['fn_stem']:i['kp'] for i in feature_map}
    i=0
    size_stats_dict_list=[]
    bins=[0,10,20,30,40]
    for fn_stem,v in tqdm(kp_dict.items()):
        kp_df=pd.DataFrame(v)
        size_stats_dict=kp_df['size'].describe().to_dict()
        size_stats_dict['num']=len(v)        
        
        size_stats_dict['fn_stem']=fn_stem
        fn_key,fn_idx=fn_stem.split("_")
        size_stats_dict['fn_key']=fn_key
        size_stats_dict['fn_idx']=fn_idx
        
        coord=coords[fn_key][int(fn_idx)]
        size_stats_dict['geometry']=Point(coord)
        # print(size_stats_dict)        
        
        # print(kp_df['size'])
        fre_size=kp_df[['size']].apply(pd.Series.value_counts,bins=bins,).to_dict()['size']
        fre_size={'{}_{}'.format(k.left,k.right):v for k,v in fre_size.items()}
        # print(fre_size)
        size_stats_dict.update(fre_size)
        
        size_stats_dict_list.append(size_stats_dict)
        # if i==10:break
        # i+=1
    kp_size_stats_df=pd.DataFrame.from_dict(size_stats_dict_list)
    wgs84=pyproj.CRS('EPSG:4326')
    kp_size_stats_gdf=gpd.GeoDataFrame(kp_size_stats_df,geometry=kp_size_stats_df.geometry,crs=wgs84) 
    return kp_size_stats_gdf   

def featureMap_stat(featureMap_gdf,feature_map_region,imgs_root,cluster_idx=0):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arrow, Circle
    from tqdm import tqdm    
    
    fn=featureMap_gdf.loc[featureMap_gdf['{}'.format(cluster_idx)].idxmax(),'fn_stem']
    # print(fn)
    print(cluster_idx,'_freq:',featureMap_gdf['{}'.format(cluster_idx)].max())
    img=get_image(os.path.join(imgs_root,fn+'.jpg'))
    # print(img.shape)
    img=img[:int(img.shape[0]*(70/100))]    
    # plt.imshow(img)
    
    kp_dict={i['fn_stem']:{'kp':i['kp'],'labels':i['labels']} for i in feature_map_region}
    kp=kp_dict[fn]['kp']
    # print(len(kp))
    
    colors=np.random.rand(32,3)
    labels=kp_dict[fn]['labels']
    # patches=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=kp[i]['size'], color='red',fill=True,alpha=0.5) if labels[i]==cluster_idx 
    #          else Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=kp[i]['size'], color='b',fill=False)
    #          for i in range(len(kp))]
    
    # patches=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=5, color='red',fill=True,alpha=0.5) if labels[i]==cluster_idx 
    #      else Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=kp[i]['size'], color='b',fill=False,alpha=0)
    #      for i in range(len(kp))]
    
    # patches=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=5, color=colors[labels[i]],fill=True,alpha=0.8) for i in range(len(kp))]    
    patches=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=kp[i]['size'], color=colors[labels[i]],fill=False,alpha=0.8) for i in range(len(kp))]    

    # fig, ax=plt.subplots(1)
    fig=plt.figure(figsize=(30,15))
    ax=fig.add_subplot(111)
    ax.imshow(img)
    for p in patches:
        ax.add_patch(p)    
    plt.show(fig)

def kp_show(feature_map,imgs_path_list,imgs_root):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arrow, Circle,Patch
    from tqdm import tqdm  
    import pandas as pd
    import matplotlib
    
    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 28}
    matplotlib.rc('font', **font)
    
    fig, axs=plt.subplots(1, 2, constrained_layout=True,figsize=(50,10))
    axs=axs.flat
    bins=[0,10,20,30,40]
    for idx,fn_stem in enumerate(imgs_path_list):        
        img=get_image(os.path.join(imgs_root,fn_stem+'.jpg'))
        # print(img.shape)
        img=img[:int(img.shape[0]*(70/100))]    
        # plt.imshow(img)   
        kp_dict={i['fn_stem']:i['kp'] for i in feature_map}
        kp=kp_dict[fn_stem]
        # print(kp)
        kp_df=pd.DataFrame.from_dict(kp)
        # print(kp_df)
        fre_size=kp_df[['size']].apply(pd.Series.value_counts,bins=bins,).to_dict()['size']
        fre_size={'{}-{}'.format(k.left,k.right):v for k,v in fre_size.items()}
        # print(fre_size)
        patches_1=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=3, color='red',fill=True,alpha=0.8) if kp[i]['size'] in range(0,10) 
                 else Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=3, color='b',fill=True,alpha=0.8) for i in range(len(kp))]
        patches_2=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]),radius=kp[i]['size'], color='red',fill=True,alpha=0.8) if kp[i]['size'] in range(0,10) 
                 else Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=kp[i]['size'], color='b',fill=False,alpha=0.8) for i in range(len(kp))]        
        
        axs[idx].imshow(img)
        for p in patches_2:
            axs[idx].add_patch(p) 
        axs[idx].set_title('key-points size:{}'.format(fre_size))
    C_1=Patch(color='red',fill=True,alpha=0.8,label='kp size in (0,10]')
    C_2=Patch(color='b',fill=False,alpha=0.8,label='kp size not in (0,10]')        
    plt.legend(handles=[C_1,C_2])    
    # plt.show()  
    plt.savefig('./graph/kp size 0-10.png',dpi=300)        
            # colors=np.random.rand(32,3)
# imgs_path_list=['320915fc28cd251103ec01b3_1','daf31942100f17f0611809f5_2'] # '36599cece30ad8582ab97cf0_3'  
# kp_show(feature_map_region,imgs_path_list,imgs_root)    
    

if __name__=="__main__":
    imgs_root='./data/panoramic imgs_tour line valid' #'./data/panoramic imgs valid'
    img_fp_list=glob.glob(os.path.join(imgs_root,'*.jpg'))
    # kp=STAR_detection(img_fp_list[460])
    
    # kmeans=feature_builder_BOW().get_visual_BOW(img_fp_list)    
    # with open('./processed data/tl_visual_BOW_region.pkl','wb') as f: # 使用with结构避免手动的文件关闭操作 './processed data/visual_BOW_region.pkl'
    #     pickle.dump(kmeans,f) #存储kmeans聚类模型
    # print("_"*50)       
    
    # with open('./processed data/tl_visual_BOW_region.pkl','rb') as f:
    #     kmeans=pickle.load(f) 
    # feature_map=feature_builder_BOW().get_feature_map(img_fp_list,kmeans)    
    # with open('./processed data/tl_feature_map_region.pkl','wb') as f: # 使用with结构避免手动的文件关闭操作
    #     pickle.dump(feature_map,f) #存储kmeans聚类模型
    
    with open('./processed data/tl_feature_map_region.pkl','rb') as f:
        feature_map_region=pickle.load(f)     
    # # a=feature_map_region[0]
    with open('./processed data/coords_tourLine.pkl','rb') as f:
        coords=pickle.load(f)  
    # featureMap_gdf=kps_desciptors_BOW_feature(feature_map_region,coords,)    
    # gpd2postSQL(featureMap_gdf,table_name='tl_featureMap',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    # featureMap_gdf=postSQL2gpd(table_name='tl_featureMap',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # featureMap_stat(featureMap_gdf,feature_map_region,imgs_root,cluster_idx=7)  

    kp_size_stats_gdf=kp_stats(feature_map_region,coords)
    gpd2postSQL(kp_size_stats_gdf,table_name='tl_kp_size_stats',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')

    #show 
    # imgs_path_list=['320915fc28cd251103ec01b3_1','daf31942100f17f0611809f5_2'] # '36599cece30ad8582ab97cf0_3'  
    # kp_show(feature_map_region,imgs_path_list,imgs_root) 