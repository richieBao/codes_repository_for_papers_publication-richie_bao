# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:27:47 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import pickle
from database import postSQL2gpd,gpd2postSQL
from segs_object_analysis import seg_equirectangular_idxs
import glob,os  
import numpy as np
import pandas as pd
from pathlib import Path

xian_epsg=32649 #Xi'an   WGS84 / UTM zone 49N
wgs84_epsg=4326

class dynamicStreetView_visualPerception:
    '''
    class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知
    
    Paras:
    imgs_fp - 图像路径列表
    knnMatch_ratio - 图像匹配比例，默认为0.75
    '''
    def __init__(self,imgs_fp,knnMatch_ratio=0.75):
        self.knnMatch_ratio=knnMatch_ratio
        self.imgs_fp=imgs_fp
    
    def kp_descriptor(self,img_fp):
        import cv2 as cv
        '''
        function - 提取关键点和获取描述子
        '''
        img=cv.imread(img_fp)
        star_detector=cv.xfeatures2d.StarDetector_create()        
        key_points=star_detector.detect(img) #应用处理Star特征检测相关函数，返回检测出的特征关键点
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #将图像转为灰度
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征
        return kp,des
        
     
    def feature_matching(self,des_1,des_2,kp_1=None,kp_2=None):
        import cv2 as cv
        '''
        function - 图像匹配
        '''
        bf=cv.BFMatcher()
        matches=bf.knnMatch(des_1,des_2,k=2)
        
        '''
        可以由匹配matches返回关键点（train,query）的位置索引，train图像的索引，及描述子之间的距离
        DMatch.distance - Distance between descriptors. The lower, the better it is.
        DMatch.trainIdx - Index of the descriptor in train descriptors
        DMatch.queryIdx - Index of the descriptor in query descriptors
        DMatch.imgIdx - Index of the train image.
        '''
        '''
        if kp_1 !=None and kp_2 != None:
            kp1_list=[kp_1[mat[0].queryIdx].pt for mat in matches]
            kp2_list=[kp_2[mat[0].trainIdx].pt for mat in matches]
            des_distance=[(mat[0].distance,mat[1].distance) for mat in matches]
            print(des_distance[:5])
        '''
        
        good=[]
        for m,n in matches:
            if m.distance < self.knnMatch_ratio*n.distance:
                good.append(m) 
        #good_num=len(good)
        return good #,good_num
    
    
    def sequence_statistics(self):
        from tqdm import tqdm
        '''
        function - 序列图像匹配计算，每一位置图像与后续所有位置匹配分析
        '''        
        des_list=[]
        print("计算关键点和描述子...")
        for f in tqdm(self.imgs_fp):        
            _,des=self.kp_descriptor(f)
            des_list.append(des)
        matches_sequence={}
        print("计算序列图像匹配数...")
        for i in tqdm(range(len(des_list)-1)):
            matches_temp=[]
            for j_des in des_list[i:]:
                matches_temp.append(self.feature_matching(des_list[i],j_des))
            matches_sequence[i]=matches_temp
        matches_num={k:[len(v) for v in val] for k,val in matches_sequence.items()}
        return matches_num  

class movingAverage_inflection:
    import pandas as pd
    
    '''
    class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点
    
    Paras:
    series - pandas 的Series格式数据
    window - 滑动窗口大小，值越大，平滑程度越大
    plot_intervals - 是否打印置信区间，某人为False 
    scale - 偏差比例，默认为1.96, 
    plot_anomalies - 是否打印异常值，默认为False,
    figsize - 打印窗口大小，默认为(15,5),
    threshold - 拐点阈值，默认为0
    '''
    def __init__(self,series, window, plot_intervals=False, scale=1.96, plot_anomalies=False,figsize=(15,5),threshold=0):
        self.series=series
        self.window=window
        self.plot_intervals=plot_intervals
        self.scale=scale
        self.plot_anomalies=plot_anomalies
        self.figsize=figsize
        
        self.threshold=threshold
        self.rolling_mean=self.movingAverage()
    
    def masks(self,vec):
        '''
        function - 寻找曲线水平和纵向的斜率变化，参考 https://stackoverflow.com/questions/47342447/find-locations-on-a-curve-where-the-slope-changes
        '''
        d=np.diff(vec)
        dd=np.diff(d)

        # Mask of locations where graph goes to vertical or horizontal, depending on vec
        to_mask=((d[:-1] != self.threshold) & (d[:-1] == -dd-self.threshold))
        # Mask of locations where graph comes from vertical or horizontal, depending on vec
        from_mask=((d[1:] != self.threshold) & (d[1:] == dd-self.threshold))
        return to_mask, from_mask
        
    def apply_mask(self,mask, x, y):
        return x[1:-1][mask], y[1:-1][mask]   
    
    def knee_elbow(self):
        '''
        function - 返回拐点的起末位置
        '''        
        x_r=np.array(self.rolling_mean.index)
        y_r=np.array(self.rolling_mean)
        to_vert_mask, from_vert_mask=self.masks(x_r)
        to_horiz_mask, from_horiz_mask=self.masks(y_r)     

        to_vert_t, to_vert_v=self.apply_mask(to_vert_mask, x_r, y_r)
        from_vert_t, from_vert_v=self.apply_mask(from_vert_mask, x_r, y_r)
        to_horiz_t, to_horiz_v=self.apply_mask(to_horiz_mask, x_r, y_r)
        from_horiz_t, from_horiz_v=self.apply_mask(from_horiz_mask, x_r, y_r)    
        return x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v

    def movingAverage(self):
        rolling_mean=self.series.rolling(window=self.window).mean()        
        return rolling_mean        

    def plot_movingAverage(self,inflection=False):
        import numpy as np
        from sklearn.metrics import median_absolute_error, mean_absolute_error
        import matplotlib.pyplot as plt
        """
        function - 打印移动平衡/滑动窗口，及拐点
        """

        plt.figure(figsize=self.figsize)
        plt.title("Moving average\n window size = {}".format(self.window))
        plt.plot(self.rolling_mean, "g", label="Rolling mean trend")

        #打印置信区间，Plot confidence intervals for smoothed values
        if self.plot_intervals:
            mae=mean_absolute_error(self.series[self.window:], self.rolling_mean[self.window:])
            deviation=np.std(self.series[self.window:] - self.rolling_mean[self.window:])
            lower_bond=self.rolling_mean - (mae + self.scale * deviation)
            upper_bond=self.rolling_mean + (mae + self.scale * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # 显示异常值，Having the intervals, find abnormal values
            if self.plot_anomalies:
                anomalies=pd.DataFrame(index=self.series.index, columns=self.series.to_frame().columns)
                anomalies[self.series<lower_bond]=self.series[self.series<lower_bond].to_frame()
                anomalies[self.series>upper_bond]=self.series[self.series>upper_bond].to_frame()
                plt.plot(anomalies, "ro", markersize=10)
                
        if inflection:
            x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v=self.knee_elbow()
            plt.plot(x_r, y_r, 'b-')
            plt.plot(to_vert_t, to_vert_v, 'r^', label='Plot goes vertical')
            plt.plot(from_vert_t, from_vert_v, 'kv', label='Plot stops being vertical')
            plt.plot(to_horiz_t, to_horiz_v, 'r>', label='Plot goes horizontal')
            plt.plot(from_horiz_t, from_horiz_v, 'k<', label='Plot stops being horizontal')     
            

        plt.plot(self.series[self.window:], label="Actual values")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.xticks(rotation='vertical')
        plt.show()

def vanishing_position_length(matches_num,coordi_df,epsg,**kwargs):
    from shapely.geometry import Point, LineString, shape
    import geopandas as gpd
    import pyproj
    '''
    function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离
    
    Paras:
    matches_num - 由类dynamicStreetView_visualPerception计算的特征关键点匹配数量
    coordi_df - 包含经纬度的DataFrame，其列名为：lon,lat
    **kwargs - 同类movingAverage_inflection配置参数
    '''
    MAI_paras={'window':15,'plot_intervals':True,'scale':1.96, 'plot_anomalies':True,'figsize':(15*2,5*2),'threshold':0}
    MAI_paras.update(kwargs)   
    #print( MAI_paras)
    
    vanishing_position={}
    for idx in range(len(matches_num)): 
        x=np.array(range(idx,idx+len(matches_num[idx]))) 
        y=np.array(matches_num[idx])
        y_=pd.Series(y,index=x)   
        MAI=movingAverage_inflection(y_, window=MAI_paras['window'],plot_intervals=MAI_paras['plot_intervals'],scale=MAI_paras['scale'], plot_anomalies=MAI_paras['plot_anomalies'],figsize=MAI_paras['figsize'],threshold=MAI_paras['threshold'])   
        _,_,_,_,from_vert_t, _,_, _,from_horiz_t,_=MAI.knee_elbow()
        if np.any(from_horiz_t!= None) :
            vanishing_position[idx]=(idx,from_horiz_t[0])
        else:
            vanishing_position[idx]=(idx,idx)
    vanishing_position_df=pd.DataFrame.from_dict(vanishing_position,orient='index',columns=['start_idx','end_idx'])
    vanishing_position_df['geometry']=vanishing_position_df.apply(lambda idx:LineString(coordi_df[idx.start_idx:idx.end_idx]['geometry'].tolist()), axis=1)

    crs_4326={'init': 'epsg:4326'}
    vanishing_position_gdf=gpd.GeoDataFrame(vanishing_position_df,geometry='geometry',crs=crs_4326)
    
    crs_=pyproj.CRS(epsg) 
    vanishing_position_gdf_reproj=vanishing_position_gdf.to_crs(crs_)
    vanishing_position_gdf_reproj['length']=vanishing_position_gdf_reproj.geometry.length
    return vanishing_position_gdf_reproj

def movingAverage(series,window):
    rolling_mean=series.rolling(window=window).mean()        
    return rolling_mean


def tourLine_segs_vanishing_position_length(tourLine_segment,img_fp_list_sorted,coordi_df,xian_epsg):
    from tqdm import tqdm
    
    vanishing_dict={}
    for k,v in tqdm(tourLine_segment.items()):
        img_fp_seg_list=img_fp_list_sorted[v[0]:v[1]]
        dsv_vp=dynamicStreetView_visualPerception(img_fp_seg_list) #[:200]
        matches_num=dsv_vp.sequence_statistics()        
        
        coordi_seg_df=coordi_df[v[0]:v[1]]
        vanishing_gpd=vanishing_position_length(matches_num,coordi_seg_df,epsg="EPSG:{}".format(xian_epsg),threshold=0)
        
        vanishing_dict[k]=vanishing_gpd
        
    with open('./processed data/tourLine_vanishing.pkl','wb') as f:
        pickle.dump(vanishing_dict,f)
    return vanishing_dict

def segs_vanishing_statistics(vanishing_dict):
    segs_Vanishing_stat={}
    for k,vanishing_gpd in vanishing_dict.items():        
        vanishing_length_desc=vanishing_gpd.length.describe()
        vanishing_fre=vanishing_gpd.length.value_counts(bins=5)
        
        segs_Vanishing_stat[k]={'vanishing_length_desc':vanishing_length_desc,'vanishing_fre':vanishing_fre}
    return segs_Vanishing_stat
    
def vanishing_segment_mark(vanishing_length,length_moving_reverse,tourLine_segment,segment_name):
    import matplotlib.pyplot as plt
    import matplotlib
    
    font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size'   : 28}
    matplotlib.rc('font', **font)
    
    fig, ax=plt.subplots(figsize=(30,10))
    ax.plot(vanishing_length,label='vanishing distance_north')
    
    length_moving_reverse_list=length_moving_reverse.to_list()
    length_moving_reverse_list.reverse()
    ax.plot(length_moving_reverse_list,'--',label='vanishing distance_south')
    
    
    v_length=[]
    for k,v in tourLine_segment.items():
        # print(k,v)
        # v_length_seg=vanishing_length.iloc[v[0]:v[1]].to_list()
        # print(length_seg)
        # print(segment_name[k])
        ax.vlines(v[1],0,800,colors='r',linestyle='--',) #label=segment_name[k]
        ax.text(v[1],920, segment_name[k], fontsize=22,rotation=-90, rotation_mode='anchor',va='top')
    plt.legend(loc=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.show()
    plt.savefig('./graph/vanishing_segment_mark.png',dpi=300)

if __name__=="__main__":
    # 01
    # tourLine_seg_path='./processed data/tour line seg'
    # tourline_label_seg_path='./processed data/tourline_label_seg'
    # tourLine_img_path='./data/panoramic imgs_tour line valid'
    # with open('./processed data/coords_tourLine.pkl','rb') as f:
    #     coords_tourLine=pickle.load(f) 
    # tourLine_panorama_object_percent_gdf=seg_equirectangular_idxs(tourline_label_seg_path,tourLine_seg_path,tourLine_img_path,coords_tourLine,)
    # gpd2postSQL(tourLine_panorama_object_percent_gdf,table_name='tourLine_panorama_object_percent',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    #02
    #sift and match    
    # img_fp_list=glob.glob(os.path.join(tourLine_img_path,'*.jpg'))
    # img_fp_dict={int(Path(p).stem.split('_')[-1]):p for p in img_fp_list}
    # img_fp_key=list(img_fp_dict.keys())
    # img_fp_key.sort()
    # img_fp_list_sorted=[img_fp_dict[k] for k in img_fp_key]
    # img_fp_list_sorted.reverse()
    
    # dsv_vp=dynamicStreetView_visualPerception(img_fp_list_sorted) #[:200]
    # matches_num=dsv_vp.sequence_statistics()

    #03
    # idx=508 
    # x=np.array(range(idx,idx+len(matches_num[idx]))) 
    # y=np.array(matches_num[idx])
    # y_=pd.Series(y,index=x)
    # MAI=movingAverage_inflection(y_, window=15,plot_intervals=True,scale=1.96, plot_anomalies=True,figsize=(15*2,5*2),threshold=0)
    # MAI.plot_movingAverage(inflection=True)
    
    #04
    # tourLine_panorama_object_percent_gdf=postSQL2gpd(table_name='tourLine_panorama_object_percent',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # coordi_df=tourLine_panorama_object_percent_gdf.sort_values(by='fn_idx')
    # vanishing_gpd=vanishing_position_length(matches_num,coordi_df,epsg="EPSG:{}".format(xian_epsg),threshold=0)
    # print("感知消失距离统计:","_"*50,"\n")
    # print(vanishing_gpd[vanishing_gpd["length"] >1].length.describe())
    # print("频数统计：","_"*50,"\n")
    # print(vanishing_gpd[vanishing_gpd["length"] >1]["length"].value_counts(bins=5))
    #'tourLine_vanishing'
    # gpd2postSQL(vanishing_gpd,table_name='tourLine_vanishing_reverse',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    
    #05
    tourLine_vanishing_gdf=postSQL2gpd(table_name='tourLine_vanishing',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    length_moving=movingAverage(tourLine_vanishing_gdf.length,window=15)
    length_moving.plot(figsize=(20,10))    
    
    tourLine_vanishing_reverse_gdf=postSQL2gpd(table_name='tourLine_vanishing_reverse',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    length_moving_reverse=movingAverage(tourLine_vanishing_reverse_gdf.length,window=15)
    length_moving_reverse.plot(figsize=(20,10))  
    
    #06
    tourLine_segment={
                0:(0,39),
                1:(39,101),
                2:(101,191),
                3:(191,290),
                4:(290,367),
                5:(367,437),
                6:(437,462),
                7:(462,488),
                8:(488,565),
                9:(565,603)
                }

    # vanishing_dict=tourLine_segs_vanishing_position_length(tourLine_segment,img_fp_list_sorted,coordi_df,xian_epsg)
    # with open('./processed data/tourLine_vanishing.pkl','rb') as f:
    #     vanishing_dict=pickle.load(f) 
    # segs_Vanishing_stat=segs_vanishing_statistics(vanishing_dict)
    # segs_vanishing_desc={k:segs_Vanishing_stat[k]['vanishing_length_desc'] for k in segs_Vanishing_stat.keys()}
  
    # # pd.set_option('display.max_columns', None)
    # segs_vanishing_desc_df=pd.DataFrame.from_dict(segs_vanishing_desc)

    # tour_line_seg_gdf=postSQL2gpd(table_name='tour_line_seg',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    tourLine_segment_name={
                0:'Jianfu Temple Road',
                1:'North section of Zhuque Street',
                2:'Friendship Road',
                3:'Changan Road',
                4:'South Gate Bends',
                5:'South Street',
                6:'Bell Tower Loop',
                7:'West Street',
                8:'Hui Street',
                9:'Xihuamen Street' }
    vanishing_segment_mark(length_moving,length_moving_reverse,tourLine_segment,tourLine_segment_name)

    