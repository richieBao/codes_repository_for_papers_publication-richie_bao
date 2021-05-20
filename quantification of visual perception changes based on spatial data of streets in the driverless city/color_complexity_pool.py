# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:21:40 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
ref:https://towardsdatascience.com/color-identification-in-images-machine-learning-application-b26e770c4c71
"""
import pickle
from database import postSQL2gpd,gpd2postSQL
import os

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    import cv2
    
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def find_dominant_colors(imgs_root,coords,resize_scale=0.5,number_of_colors=10,show_chart=False):
    import glob,os  
    from tqdm import tqdm
    import cv2
    from sklearn.cluster import KMeans
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pandas as pd
    from shapely.geometry import Point
    import pyproj
    import geopandas as gpd
    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))#[-133:]
    # print(img_fns)
    img_dominant_color=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    i=0
    for fn in tqdm(img_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")   
        print('\n',fn_stem)
        
        img=get_image(fn)
        # print(img.shape)
        img_h,img_w,_=img.shape
        modified_img=cv2.resize(img, (int(img_w*resize_scale),int(img_h*resize_scale),), interpolation = cv2.INTER_AREA)
        # print(modified_img.shape)
        modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
        # print(modified_img)
        clf=KMeans(n_clusters=number_of_colors)
        labels=clf.fit_predict(modified_img)
        # print(np.unique(labels))
        
        counts=Counter(labels)
        print(counts)
        center_colors=clf.cluster_centers_
        ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
        # print(ordered_colors)
        hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors=[ordered_colors[i] for i in counts.keys()]
        
        if (show_chart):
            plt.figure(figsize = (8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)       
        
        coord=coords[fn_key][int(fn_idx)]
        color_dic={k:hex_colors[k] for k in range(number_of_colors) }    
        color_dic.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})    
        img_dominant_color=img_dominant_color.append(color_dic,ignore_index=True)        
        
        # break
        if i==2:break
        i+=1
        
    wgs84=pyproj.CRS('EPSG:4326')
    img_dominant_color_gdf=gpd.GeoDataFrame(img_dominant_color,geometry=img_dominant_color.geometry,crs=wgs84) 
    
    return img_dominant_color_gdf


#pool
with open('./processed data/coords_tourLine.pkl','rb') as f: #'./processed data/coords.pkl'
    coords=pickle.load(f)
resize_scale=0.5
number_of_colors=16
show_chart=False

def find_dominant_colors_pool(fn):
    import glob,os  
    from tqdm import tqdm
    import cv2
    from sklearn.cluster import KMeans
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pandas as pd
    from shapely.geometry import Point
    import pyproj
    import geopandas as gpd
    # print("_"*50)
    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")   
    # print(fn_stem,fn_key,fn_idx)
    
    img=get_image(fn)
    # print(img.shape)
    img_h,img_w,_=img.shape
    modified_img=cv2.resize(img, (int(img_w*resize_scale),int(img_h*resize_scale),), interpolation = cv2.INTER_AREA)
    # print(modified_img.shape)
    modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    # print(modified_img)
    clf=KMeans(n_clusters=number_of_colors)
    labels=clf.fit_predict(modified_img)
    # print(np.unique(labels))
    
    counts=Counter(labels)
    # print(counts)
    center_colors=clf.cluster_centers_
    ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
    # print(ordered_colors)
    hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors=[ordered_colors[i] for i in counts.keys()]
    
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)       

    coord=coords[fn_key][int(fn_idx)]
    color_dic={k:hex_colors[k] for k in range(number_of_colors) }    
    color_dic.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})    
    
    return color_dic

def dominant2cluster_colors(imgs_root,coords,resize_scale=0.5,number_of_colors=10,show_chart=False):
    import glob,os  
    from tqdm import tqdm
    import cv2
    from sklearn.cluster import KMeans
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pandas as pd
    from shapely.geometry import Point
    import pyproj
    import geopandas as gpd
    import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg   
    
    img_fns=glob.glob(os.path.join(imgs_root,'*.jpg'))#[-133:]
    # print(img_fns)
    img_dominant_color=pd.DataFrame(columns=["fn_stem","fn_key","fn_idx","geometry",]+list(range(number_of_colors)))
    i=0
    for fn in tqdm(img_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")   
        print('\n',fn_stem)
        
        img=get_image(fn)
        # print(img.shape)
        img_h,img_w,_=img.shape
        modified_img_w,modified_img_h=int(img_w*resize_scale),int(img_h*resize_scale),
        modified_img=cv2.resize(img, (modified_img_w,modified_img_h), interpolation = cv2.INTER_AREA)
        # print(modified_img.shape)
        modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
        # print(modified_img.shape)
        clf=KMeans(n_clusters=number_of_colors)
        labels=clf.fit_predict(modified_img)
        center_colors=clf.cluster_centers_
        # print(np.unique(labels))
        # print(labels.shape)
        # print(modified_img.shape)
        # print(modified_img_w,modified_img_h)
        labels_RGB=np.array([center_colors[i] for i in labels])
        # print(labels_RGB.shape)
        labels_RGB_restore=labels_RGB.reshape((modified_img_h,modified_img_w,3))
        # print(labels_RGB_restore.shape)        
        plt.imshow(labels_RGB_restore/255)
        plt.show()
        
        labels_restored=labels.reshape((modified_img_h,modified_img_w,))
        plt.imshow(labels_restored,cmap="gist_ncar")      
        plt.show()
        
        # counts=Counter(labels)
        # print(counts)       
        # ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
        # print(ordered_colors)
        # hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        # rgb_colors=[ordered_colors[i] for i in counts.keys()]
        
        if (show_chart):
            plt.figure(figsize = (8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)   
            plt.show()
            
        # from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
        # from skimage.segmentation import mark_boundaries
        # from skimage.util import img_as_float
        # # print(labels_restore)
        # segments_quick = quickshift(img_as_float(labels_RGB_restore/255), kernel_size=1, max_dist=1, ratio=0.5)
        # print(len(np.unique(segments_quick)))
        # plt.imshow(segments_quick,cmap="gist_ncar")
        # plt.show()
        
        # fig, ax = plt.subplots(2, 2, figsize=(40, 40), sharex=True, sharey=True)
        # plt.imshow(mark_boundaries(labels_RGB_restore/255, segments_quick))
        # for a in ax.ravel():
        #     a.set_axis_off()        
        # plt.tight_layout()
        # plt.show()
        
        # from scipy.ndimage import measurements
        # lw, num = measurements.label(labels_restored)
        # print(lw,lw.shape)
        # print(num)
        
        from skimage import measure
        img_labeled = measure.label(labels_restored, connectivity=1)
        # print(img_labeled)
        # Get the indices for each region, excluding zeros
        # idx = [np.where(img_labeled == label) for label in np.unique(img_labeled) if label]   
        # print(img_labeled.shape,len(np.unique(img_labeled)))
        # # Get the bounding boxes of each region (ignoring zeros)
        # bboxes = [area.bbox for area in measure.regionprops(img_labeled)]
        plt.imshow(img_labeled,cmap="gist_ncar")
        plt.show()

        
        # coord=coords[fn_key][int(fn_idx)]
        # color_dic={k:hex_colors[k] for k in range(number_of_colors) }    
        # color_dic.update({"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord)})    
        # img_dominant_color=img_dominant_color.append(color_dic,ignore_index=True)       
        
        # counts=Counter(img_labeled)
        # print(counts)
        
        
        # break
        if i==0:break
        i+=1
        
    # wgs84=pyproj.CRS('EPSG:4326')
    # img_dominant_color_gdf=gpd.GeoDataFrame(img_dominant_color,geometry=img_dominant_color.geometry,crs=wgs84) 
    
    # return img_dominant_color_gdf
    
    
#pool
# with open('./processed data/coords.pkl','rb') as f:
#     coords=pickle.load(f)
# resize_scale=0.1
# number_of_colors=16
# show_chart=False    
def dominant2cluster_colors_pool(fn):
    import glob,os  
    from tqdm import tqdm
    import cv2
    from sklearn.cluster import KMeans
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pandas as pd
    from shapely.geometry import Point
    import pyproj
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from skimage import measure
    
    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")
    print(fn_stem)
    
    img=get_image(fn)
    # print(img.shape)
    img=img[:int(img.shape[0]*(70/100))]
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()

    img_h,img_w,_=img.shape
    modified_img_w,modified_img_h=int(img_w*resize_scale),int(img_h*resize_scale),
    modified_img=cv2.resize(img, (modified_img_w,modified_img_h), interpolation = cv2.INTER_AREA)
    print(modified_img.shape)
    modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    clf=KMeans(n_clusters=number_of_colors)
    labels=clf.fit_predict(modified_img)
    center_colors=clf.cluster_centers_
    
    labels_restored=labels.reshape((modified_img_h,modified_img_w,))    
    img_labeled=measure.label(labels_restored, connectivity=1)
    # plt.imshow(img_labeled,cmap="gist_ncar")
    # plt.show()
    
    # print(img_labeled)
    counts=Counter(img_labeled.flatten())
    # print(counts)
    coord=coords[fn_key][int(fn_idx)]
    color_dic={"fn_stem":fn_stem,"fn_key":fn_key,"fn_idx":int(fn_idx),"geometry":Point(coord),'counter':dict(counts)} 
    # print(color_dic)

    return color_dic

def dominant2cluster_colors_imshow(fn,coords,resize_scale=0.1,number_of_colors=16):
    import glob,os  
    from tqdm import tqdm
    import cv2
    from sklearn.cluster import KMeans
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pandas as pd
    from shapely.geometry import Point
    import pyproj
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from skimage import measure
    import matplotlib
    
    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)      
    
    
    fig, axs=plt.subplots(1, 4, figsize=(30, 8))
    
    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")
    print(fn_stem)
    
    img=get_image(fn)
    # print(img.shape)
    img=img[:int(img.shape[0]*(70/100))]
    # print(img.shape)
    axs[0].imshow(img)
    axs[0].set_title('Panorama') 
    axs[0].set_ylabel('color richness index:90.204') #64.178

    img_h,img_w,_=img.shape
    modified_img_w,modified_img_h=int(img_w*resize_scale),int(img_h*resize_scale),
    modified_img=cv2.resize(img, (modified_img_w,modified_img_h), interpolation = cv2.INTER_AREA)
    # print(modified_img.shape)
    modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    clf=KMeans(n_clusters=number_of_colors)
    labels=clf.fit_predict(modified_img)
    center_colors=clf.cluster_centers_
    
    labels_RGB=np.array([center_colors[i] for i in labels])
    labels_RGB_restore=labels_RGB.reshape((modified_img_h,modified_img_w,3))
    axs[1].imshow(labels_RGB_restore/255)
    axs[1].set_title('Theme color distribution') 
    
    labels_restored=labels.reshape((modified_img_h,modified_img_w,))    
    img_labeled=measure.label(labels_restored, connectivity=1)
    axs[3].imshow(img_labeled,cmap="gist_ncar")
    axs[3].set_title('Theme color proximity clustering')
    
    counts=Counter(labels)    
    ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
    hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors=[ordered_colors[i] for i in counts.keys()]

    axs[2].pie(counts.values(), labels=hex_colors, colors=hex_colors,rotatelabels =True,radius=0.5)   #labels=hex_colors,
    axs[2].set_title('Theme color') 
    
    fig.tight_layout() 
    # plt.show()
    plt.savefig('./graph/theme color cluster.png',dpi=300)
    
    

if __name__=="__main__":
    # with open('./processed data/coords.pkl','rb') as f:
    #     coords=pickle.load(f) 
    # img_path=r'./data/panoramic imgs valid'
    # img_dominant_color_gdf=find_dominant_colors(img_path,coords,number_of_colors=32,show_chart=True)
    # gpd2postSQL(img_dominant_color_gdf,table_name='img_dominant_color',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # pass

    # dominant2cluster_colors(img_path,coords,resize_scale=0.1,number_of_colors=16,show_chart=True)
    # dominant2cluster_colors_pool(os.path.join(img_path,'b0187d856514af4a29201a3a_0.jpg'))
    
    # dominant2cluster_colors_imshow(os.path.join(img_path,'320915fc28cd251103ec01b3_0.jpg'),coords)
    
    pass