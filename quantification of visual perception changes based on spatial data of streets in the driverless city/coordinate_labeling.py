# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:42:44 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
import glob,os


def label_panorama(panorama_fn,polar_fn,cube_fn,sphere_fn):
    from PIL import Image,ImageOps
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import matplotlib
    
    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 28}
    matplotlib.rc('font', **font)
    title_fontsize=55
    
    # vertical_label_radians=np.linspace(0, np.pi,14)
    # vertical_label_degree=["{:.2f}".format(90-math.degrees(radi)) for radi in vertical_label_radians]
    vertical_label_degree=90-np.linspace(0, 180,17)
    horizontal_label_degree=180-np.linspace(0, 360,17)    
    print(horizontal_label_degree)
    
    pano=np.asarray(Image.open(panorama_fn))
    # pano=cv2.imread(panorama_fn)
    pano_height,pano_width,_=pano.shape
    print(pano_width,pano_height)
    
    # fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=(20,10))
    fig=plt.figure(figsize=(20,10))
    
    #01-pano
    # ax1=plt.subplot(131, frameon=False)  
    ax1_coords=[0, 0, 1, 1]
    ax1=fig.add_axes(ax1_coords)    
    
    ax1.imshow(pano,)     #extent=[x0, x1, y0, y1]  interpolation='bilinear',aspect='auto',origin='lower',
    ax1.set_yticks(np.linspace(0,pano_height,len(vertical_label_degree)))    
    ax1.set_yticklabels(vertical_label_degree) # provide name to the x axis tick marks   
    ax1.set_xticks(np.linspace(0,pano_width,len(horizontal_label_degree)))    
    ax1.set_xticklabels(horizontal_label_degree) # provide name to the x axis tick marks   
    ax1.axhline(y=pano_height/2,color='gray',linestyle='-.',linewidth=1)
    ax1.axvline(x=pano_width/2,color='gray',linestyle='-.',linewidth=1)
    ax1.set_title("Equirectangular format", va = 'bottom',fontsize=title_fontsize)

    #02-polar
    # polar=cv2.imread(polar_fn)
    polar=np.asarray(Image.open(polar_fn))
    polar_height,polar_width,_=polar.shape   
    ax2_coords = [0.8, 0, 1, 1]
    
    ax2_image = fig.add_axes(ax2_coords)
    ax2_image.imshow(polar, alpha = 1)
    ax2_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image    

    theta = np.linspace(0, 2 * np.pi, 73)       
    ax2_polar = fig.add_axes(ax2_coords, projection = 'polar')
    ax2_polar.patch.set_alpha(0)
    ax2_polar.set_ylim(30, 41)
    ax2_polar.set_yticks(np.arange(30, 41, 2))
    ax2_polar.set_yticklabels([])
    ax2_polar.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax2_polar.grid(True)
    ax2_polar.set_title("Polar format", va = 'bottom',fontsize=title_fontsize)
    

    #03-cube
    ax3_coords=[1.47, 0, 1, 1]
    ax3=fig.add_axes(ax3_coords)
    # cube=cv2.imread(cube_fn)
    cube=np.asarray(Image.open(cube_fn))
    cube_height,cube_width,_=cube.shape
    ax3.imshow(cube, alpha = 1)
    ax3.axis('off')    
    ax3.axvline(x=cube_width*(1/3),color='gray',linestyle='-.',linewidth=1)
    ax3.axvline(x=cube_width*(2/3),color='gray',linestyle='-.',linewidth=1)    
    ax3.set_title("Cubic format", va = 'bottom',fontsize=title_fontsize)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.4)
    ax3.text(cube_width*(1/6), cube_height*(1/4), "Top", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(3/6), cube_height*(1/4), "Back", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(5/6), cube_height*(1/4), "Down", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(1/6), cube_height*(3/4), "Left", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(3/6), cube_height*(3/4), "Front", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(5/6), cube_height*(3/4), "Right", ha="center", va="center", size=40,bbox=bbox_props)
    
    
    #04-sphere
    ax4_coords=[2.15, 0, 1, 1]
    ax4=fig.add_axes(ax4_coords)    
    # sphere=cv2.imread(sphere_fn)    
    sphere=np.asarray(Image.open(sphere_fn))
    ax4.imshow(sphere)     
    ax4.axis('off') 
    ax4.set_title("Spherical format", va = 'bottom',fontsize=title_fontsize)
    
    fig.tight_layout()
    
    # fig.savefig('./graph/preprocessed data_01',dpi=300)
    plt.show()

if __name__=="__main__":
    fn=r'1e32241a3f2fe5a796ec847f_12.jpg'
    img_panorama_fn=os.path.join('./data/panoramic imgs valid/',fn)
    img_polar_fn=os.path.join('./processed data/polar_img',fn)
    img_cube_fn=os.path.join('./processed data/img_cube',fn)
    img_sphere_fn=os.path.join('./processed data/img_sphere',fn)
    
    seg_panorama_fn=os.path.join('./processed data/img_seg',fn)
    seg_polar_fn=os.path.join('./processed data/polar_seg',fn)
    seg_cube_fn=os.path.join('./processed data/img_seg_cube',fn)
    seg_sphere_fn=os.path.join('./processed data/seg_sphere',fn)
    
    # label_panorama(img_panorama_fn,img_polar_fn,img_cube_fn,img_sphere_fn)
    label_panorama(seg_panorama_fn,seg_polar_fn,seg_cube_fn,seg_sphere_fn)