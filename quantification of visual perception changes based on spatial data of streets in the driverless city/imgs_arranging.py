# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:44:51 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from database import postSQL2gpd,gpd2postSQL

#0-15-25-50-100
img_fns={
    'vegetation':[['75245f2d4aed015a5fd4feb1_8','e88007042407fd57dd1e523a_0'],
		        ['033aa8624525bbe9c5f50dda_2','aa9dda69d2a493b92a06ce3a_1'],
				['c67fdf4bcf598975e8e582b1_11','36599cece30ad8582ab97cf0_3'],
				['b8866242b45bfb1ff80cad76_24','edbfbdf04a242554326212f5_6']],    
    
    'sky':[['ec02702c6c3b0e5e09ccddf0_15','36599cece30ad8582ab97cf0_2'],
		        ['c488235565ebb27fcd04efb1_1','73bb194cf5cb2cfcc236443a_0'],
				['2a8cd76df35285eb8515417d_32','dad50595f02890440969d9c9_2'],
				['b69c7b856514af4a28201ab6_23','c71309f0f7fb2fddbbdf8ff7_15']], 
    
    'sky_vegettion':[['415741155551fea63786f4d7_2','56023b6835284eacd618b780_2'],
   		            ['ddcf30f710b5a7a511882493_2','cd0533f0f7fb2fddbbdf8fc5_0'],
   					['3426adf70958ecccda1318f7_19','f293941cfc56794e0582c2fb_3'],
   					['91df45f82d1625bed6b74eb0_18','0c15a38935dc0df92eb96d22_4']],  
    
    'ground':[['9beb8e887af77d1e8c7727fa_1','5317936ea1ee78e232169bdd_28'],
				  ['f96b591ea8a6e39141e1924c_1','a5f8bae0792cf934912ca3f6_2'],
				  ['db0318a07af2be0baf080fd0_2','b605ba58bdeb6544e0110382_3'],
				  ['61f2074cf5cb2cfcc23644fd_0','51eb85fc0e4b738514b54bb1_0']], 
    
    'equilibrium degree':[['9beb8e887af77d1e8c7727fa_1','fe0bb711c7dc4b206da3ccf0_5'],
		         ['f41ec51774e4f36840d63e4c_36','3e4c285cb209994de7e9653d_0'], 
                 ['91df45f82d1625bed6b74eb0_18','36599cece30ad8582ab97cf0_2'],
				 ['1e32241a3f2fe5a796ec847f_27','0d70191440a0a7beacee7ab0_0']]        
    }

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

def imgs_arranging(imgs_root,img_fns):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image,ImageOps
    import matplotlib
    
    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 28}
    matplotlib.rc('font', **font)    
    
    rows=['(-0.001, 15.0]','(15.0, 25.0]','(25.0, 50.0]','(50.0, 100.0]']
    img_list=[]
    for k,fns in img_fns.items():
        img_list.append([os.path.join(imgs_root,i[1]+'.jpg') for i in fns])
    img_list=list(map(list,zip(*img_list)))
    
    img_list=flatten_lst(img_list)
    nrows_ncols=(len(list(img_fns.values())[0]),len(img_fns.keys()),)
    print(nrows_ncols)
    fig=plt.figure(figsize=(30.+1, 20.-3))
    grid=ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                   axes_pad=0.05,  # pad between axes in inch.
                     )    
    i=0
    for ax,im_fn in zip(grid,img_list):
        # print(ax)
        ax.imshow(Image.open(im_fn))           
        # ax.axis('off')         
        if i<5:ax.set_title(list(img_fns.keys())[i])        
        if i%5==0:ax.set_ylabel(rows[i//5], )
        i+=1
    fig.tight_layout()
    # plt.show()
    plt.savefig('./graph/object_idx_cube.png',dpi=300)

skyline_idxes_domain={
    'PA':[0,58,400], #400
    'SI':[0,1.9,4.3], #4.3
    'FD':[0,1.061,1.153] #1.153
    }

def skyline_idxes_imgs_arranging(idxes_df,imgs_root,domain,idx='PA'):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image,ImageOps
    import matplotlib
    
    idx_domain=domain[idx]
    print(idx_domain)
    if idx=='PA':
        lower_df=idxes_df[idxes_df.perimeter_area_ratio_mn<idx_domain[1]-20]
        upper_df=idxes_df[idxes_df.perimeter_area_ratio_mn>=idx_domain[1]+140]
        title_n='Perimeter area ratio'
    elif idx=='SI':
        lower_df=idxes_df[idxes_df.shape_index_mn<idx_domain[1]-0.5]
        upper_df=idxes_df[idxes_df.shape_index_mn>=idx_domain[1]+1]
        title_n='Shape index'
    elif idx=='FD':
        lower_df=idxes_df[idxes_df.fractal_dimension_mn<idx_domain[1]-0.01]
        upper_df=idxes_df[idxes_df.fractal_dimension_mn>=idx_domain[1]+0.01]  
        title_n='Fractal dimension'
        
    lower_fns=lower_df.fn_stem.sample(n=10)
    upper_fns=upper_df.fn_stem.sample(n=10)        

    lower_upper_fns=lower_fns.append(upper_fns)
    lower_upper_fns=[os.path.join(imgs_root,i+'.jpg') for i in lower_upper_fns]
    # print(lower_upper_fns)  

    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 26}
    matplotlib.rc('font', **font) 
    
    nrows_ncols=(2,10)
    fig=plt.figure(figsize=(30, 7.))
    grid=ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                   axes_pad=0.0,  # pad between axes in inch.
                     )      
    i=0
    rows=['({},{}]'.format(idx_domain[0],idx_domain[1]),'({},{}]'.format(idx_domain[1],idx_domain[2])]
    for ax,im_fn in zip(grid,lower_upper_fns):
        # print(ax)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_frame_on(False)
        ax.imshow(Image.open(im_fn),cmap='Greys',  interpolation='nearest')  
        if i==1:ax.set_title(title_n)   
        if i%10==0:ax.set_ylabel(rows[i//10], )
        i+=1
        
    # plt.title(title_n)      
    fig.tight_layout()    
    # plt.show()        
    plt.savefig('./graph/{}.png'.format(title_n),dpi=300)
    
def idxes_clustering_imgs_arranging(imgs_root,idxes_clustering_gdf,sample_num=3):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image,ImageOps
    import matplotlib
    
    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 28}
    matplotlib.rc('font', **font) 
    
    cluster_unique=idxes_clustering_gdf.clustering.unique()
    cluster_unique.sort()
    # print(cluster_unique)
    
    sample_total=[]
    for cluster in cluster_unique:
        sample=idxes_clustering_gdf[idxes_clustering_gdf.clustering.eq(cluster_unique[0])].sample(sample_num).fn_stem.to_list()
        sample_fn=[os.path.join(imgs_root,fn+'.jpg') for fn in sample]
        sample_total.append(sample_fn)
    # print(sample_total)
    sample_total_T=list(map(list,zip(*sample_total)))
    sample_total_T_flatten=flatten_lst(sample_total_T)
    # print(sample_total_T)
    
    fig=plt.figure(figsize=(30.+1, 20.-3))
    nrows_ncols=(sample_num,len(cluster_unique))
    grid=ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                   axes_pad=0.05,  # pad between axes in inch.
                     )  
    
    i=0
    for ax,im_fn in zip(grid,sample_total_T_flatten):
        # print(ax)
        ax.imshow(Image.open(im_fn))           
        # ax.axis('off')         
        # if i<5:ax.set_title(list(img_fns.keys())[i])        
        # if i%5==0:ax.set_ylabel(rows[i//5], )
        i+=1
    fig.tight_layout()
    plt.show()    
    

if __name__=="__main__":
    #A-object percentage
    # imgs_root='./processed data/img_cube'
    imgs_root='./data/panoramic imgs valid'
    # imgs_arranging(imgs_root,img_fns)
    
    #B-sky landscape index
    # imgs_root='./processed data/polar_sky'
    # sky_class_level_metrics_gdf=postSQL2gpd(table_name='sky_class_level_metrics',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    # skyline_idxes_imgs_arranging(sky_class_level_metrics_gdf,imgs_root,skyline_idxes_domain,idx='FD')
    
    
    #C-factors clustering
    # imgs_root='./processed data/img_cube'
    idxes_clustering_gdf=postSQL2gpd(table_name='idxes_clustering_8',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='streetscape_GSV')
    idxes_clustering_imgs_arranging(imgs_root,idxes_clustering_gdf,sample_num=8)
    