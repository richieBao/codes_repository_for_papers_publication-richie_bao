# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:45:02 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
"""
from mayavi import mlab
from tvtk.api import tvtk # python wrappers for the C++ vtk ecosystem

import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
import matplotlib.pyplot as plt # only for manipulating the input image
import glob,os, pickle

label_mapping={
       0:"pole",
       1:"slight",
       2:"bboard",
       3:"tlight",
       4:"car",
       5:"truck",
       6:"bicycle",
       7:"motor",
       8:"bus",
       9:"tsignf",
       10:"tsignb",
       11:"road",
       12:"sidewalk",
       13:"curbcut",
       14:"crosspln",
       15:"bikelane",
       16:"curb",
       17:"fence",
       18:"wall",
       19:"building",
       20:"person",
       21:"rider",
       22:"sky",
       23:"vege",
       24:"terrain",
       25:"markings",
       26:"crosszeb",
       27:"Nan",                           
       }
label_color={
    0:(117,115,102), #"pole",
    1:(212,209,156),#"slight",
    2:(224,9,9),#"bboard",
    3:(227,195,66),#"tlight",
    4:(137,147,169),#"car",
    5:(53,67,98),#"truck",
    6:(185,181,51),#"bicycle",
    7:(238,108,91),#"motor",
    8:(247,5,5),#"bus",
    9:(127,154,82),#"tsignf",
    10:(193,209,167),#"tsignb",
    11:(82,83,76),#"road",
    12:(141,142,133),#"sidewalk",
    13:(208,212,188),#"curbcut",
    14:(98,133,145),#"crosspln",
    15:(194,183,61),#"bikelane",
    16:(141,139,115),#"curb",
    17:(157,186,133),#"fence",
    18:(114,92,127),#"wall",
    19:(78,61,76),#"building",
    20:(100,56,67),#"person",
    21:(240,116,148),#"rider",
    22:(32,181,191),#"sky",
    23:(55,204,26),#"vege",
    24:(84,97,82),#"terrain",
    25:(231,24,126),#"markings",
    26:(141,173,166),#"crosszeb",
    27:(0,0,0),#"Nan",                
    }

def auto_sphere(image_file):
    # create a figure window (and scene)
    fig = mlab.figure(size=(600, 600))

    # load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    # print(texture)
    # (interpolate for a less raster appearance when zoomed in)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 1
    Nrad = 180

    # create the sphere source with a given radius and angular resolution
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,
                                       phi_resolution=Nrad)

    # print(sphere)
    # assemble rest of the pipeline, assign texture    
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)
    
    mlab.show()

def manual_sphere(image_file):
    # caveat 1: flip the input image along its first axis
    img = plt.imread(image_file) # shape (N,M,3), flip along first dim
    outfile = image_file.replace('.jfif', '_flipped.jpg')
    # flip output along first dim to get right chirality of the mapping
    img = img[::-1,...]
    plt.imsave(outfile, img)
    image_file = outfile  # work with the flipped file from now on

    # parameters for the sphere
    R = 1 # radius of the sphere
    Nrad = 180 # points along theta and phi
    phi = np.linspace(0, 2 * np.pi, Nrad)  # shape (Nrad,)
    theta = np.linspace(0, np.pi, Nrad)    # shape (Nrad,)
    phigrid,thetagrid = np.meshgrid(phi, theta) # shapes (Nrad, Nrad)

    # compute actual points on the sphere
    x = R * np.sin(thetagrid) * np.cos(phigrid)
    y = R * np.sin(thetagrid) * np.sin(phigrid)
    z = R * np.cos(thetagrid)

    # create figure
    mlab.figure(size=(600, 600))

    # create meshed sphere
    mesh = mlab.mesh(x,y,z)
    mesh.actor.actor.mapper.scalar_visibility = False
    mesh.actor.enable_texture = True  # probably redundant assigning the texture later

    # load the (flipped) image for texturing
    img = tvtk.JPEGReader(file_name=image_file)
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=0, repeat=0)
    # print(texture)
    mesh.actor.actor.texture = texture

    # tell mayavi that the mapping from points to pixels happens via a sphere
    mesh.actor.tcoord_generator_mode = 'sphere' # map is already given for a spherical mapping
    cylinder_mapper = mesh.actor.tcoord_generator
    # caveat 2: if prevent_seam is 1 (default), half the image is used to map half the sphere
    cylinder_mapper.prevent_seam = 0 # use 360 degrees, might cause seam but no fake data
    #cylinder_mapper.center = np.array([0,0,0])  # set non-trivial center for the mapping sphere if necessary

def mpl_sphere(image_file):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    img = plt.imread(image_file)
    # define a grid matching the map size, subsample along with pixels
    theta = np.linspace(0, np.pi, img.shape[0])
    phi = np.linspace(0, 2*np.pi, img.shape[1])
    
    print(img.shape)
    print(theta.shape)
    print(phi.shape)

    #'''
    count =180 #180 # keep 180 points along theta and phi
    theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
    phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
    # print(theta_inds)
    
    theta = theta[theta_inds]
    phi = phi[phi_inds]
    print(theta.shape)
    print(phi.shape)    
    
    img = img[np.ix_(theta_inds, phi_inds)]
    print("_"*50)
    print(img.shape)
    #'''

    theta,phi = np.meshgrid(theta, phi)
    print(theta.shape,phi.shape)
        
    R = 1
    # sphere
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # create 3d Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1) # we've already pruned ourselves

    # make the plot more spherical
    ax.axis('scaled')
    plt.show()

def spherical_segs_pts_show(label_seg_fn,label_color):
    from tqdm import tqdm    
    import pickle
    import numpy as np
    from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
    from PIL import Image,ImageOps
    
    fig=mlab.figure(size=(600, 600))    
    print(label_seg_fn)
    with open(label_seg_fn,'rb') as f:
        label_seg=pickle.load(f).numpy()             
    print('\nseg shape={}'.format(label_seg.shape))
    
    # define a grid matching the map size, subsample along with pixels
    theta=np.linspace(0, np.pi, label_seg.shape[0])
    phi=np.linspace(0, 2*np.pi, label_seg.shape[1])        
    print("theta shape={};phi shape={}".format(theta.shape,phi.shape))        
    theta,phi=np.meshgrid(theta, phi)
    print("theta shape={};phi shape={}".format(theta.shape,phi.shape))
    
    label_seg_color=np.array([label_color[v] for v in label_seg.flatten()]).reshape((label_seg.shape[0],label_seg.shape[1],3))
    print("\nlabel_seg_color shape={}".format(label_seg_color.shape))    

    R=10
    # sphere
    x=R * np.sin(theta) * np.cos(phi)
    y=R * np.sin(theta) * np.sin(phi)
    z=R * np.cos(theta)                
    print("x,y,z shape={},{},{}".format(x.shape,y.shape,z.shape))       
    mask=label_seg==22
    # print(len(np.extract(mask,x.T)),len(np.extract(mask,y.T)),len(np.extract(mask,z.T)),len(np.extract(mask,label_seg_color[:,:,0]/255)))
    mlab.points3d(x.T, y.T, z.T, label_seg_color[:,:,0]/255,) #opacity=0.75,scale_factor=0.1        
    # mlab.points3d(np.extract(mask,x.T),np.extract(mask,y.T),np.extract(mask,z.T),)        
    theta_phi=np.dstack((theta,phi))   
    mlab.show()
    
def spherical_segs_object_changing(label_seg_path,label_color):
    from tqdm import tqdm
    import glob,os 
    import pickle
    import numpy as np
    from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
    from PIL import Image,ImageOps
    
    # fig=mlab.figure(size=(600, 600))    
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    # print(label_seg_fns)
    for label_seg_fn in tqdm(label_seg_fns):
        print(label_seg_fn)
        with open(label_seg_fn,'rb') as f:
            label_seg=pickle.load(f).numpy()             
        print('\nseg shape={}'.format(label_seg.shape))
        
        # define a grid matching the map size, subsample along with pixels
        theta=np.linspace(0, np.pi, label_seg.shape[0])
        phi=np.linspace(0, 2*np.pi, label_seg.shape[1])        
        print("theta shape={};phi shape={}".format(theta.shape,phi.shape))        
        theta,phi=np.meshgrid(theta, phi)
        print("theta shape={};phi shape={}".format(theta.shape,phi.shape))
        
        label_seg_color=np.array([label_color[v] for v in label_seg.flatten()]).reshape((label_seg.shape[0],label_seg.shape[1],3))
        print("\nlabel_seg_color shape={}".format(label_seg_color.shape))    
    
        R=10
        # sphere
        x=R * np.sin(theta) * np.cos(phi)
        y=R * np.sin(theta) * np.sin(phi)
        z=R * np.cos(theta)                
        print("x,y,z shape={},{},{}".format(x.shape,y.shape,z.shape))       
        mask=label_seg==22
        # print(len(np.extract(mask,x.T)),len(np.extract(mask,y.T)),len(np.extract(mask,z.T)),len(np.extract(mask,label_seg_color[:,:,0]/255)))
        # mlab.points3d(x.T, y.T, z.T, label_seg_color[:,:,0]/255,) #opacity=0.75,scale_factor=0.1  
        # mlab.show()
        # mlab.points3d(np.extract(mask,x.T),np.extract(mask,y.T),np.extract(mask,z.T),)        
        theta_phi=np.dstack((theta,phi))   
        
        break    
    
def fns_sort(fns_list):
    from pathlib import Path  
    
    fns_dict={int(Path(p).stem.split('_')[-1]):p for p in fns_list}
    fns_dict_key=list(fns_dict.keys())
    fns_dict_key.sort()
    fns_dict_sorted=[fns_dict[k] for k in fns_dict_key]    
    return fns_dict_sorted       
    
def panorama_object_change(label_seg_path,label_color):
    from tqdm import tqdm
    import glob,os 
    import pickle
    import numpy as np
    from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
    from PIL import Image,ImageOps
    from pathlib import Path    
    import pandas as pd
    from sklearn import preprocessing
  
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    label_seg_fns_sorted=fns_sort(label_seg_fns)   
    
    pixels={}
    # i=0
    for label_seg_fn in tqdm(label_seg_fns_sorted):
        # print(label_seg_fn)
        with open(label_seg_fn,'rb') as f:
            label_seg=pickle.load(f).numpy()             
        # print('\nseg shape={}'.format(label_seg.shape))   
        fn_stem=Path(label_seg_fn).stem
        fn_key,fn_idx=fn_stem.split("_")        
        pixels[fn_stem]=label_seg.flatten()        
        
        # if i==10:break        
        # i+=1
        
    img_pixels_df=pd.DataFrame.from_dict(pixels,orient='index')
    pixels_diff=img_pixels_df.diff()
    pixels_diff[pixels_diff!=0]=1
    # print(img_pixels_df)
    pixels_diff_sum=pixels_diff.sum(axis=0)
    pixels_diff_array=np.array(pixels_diff_sum).reshape(label_seg.shape)
    
    min_max_scaler=preprocessing.MinMaxScaler()
    pixels_diff_array_standardization=min_max_scaler.fit_transform(pixels_diff_array)
    img_object_change=Image.fromarray(np.uint8(pixels_diff_array_standardization * 255) , 'L')
    
    img_object_change.save('./processed data/img_object_change.jpg')
    with open('./processed data/pixels_diff_array_standardization.pkl','wb') as f:
        pickle.dump(pixels_diff_array_standardization,f)    
    with open('./processed data/pixels_diff_array.pkl','wb') as f:
        pickle.dump(pixels_diff_array,f)          
    
    return img_object_change,pixels_diff_array_standardization
    
def spherical_img_pts_show(panorama_fn,FOV=False):
    from tqdm import tqdm    
    import pickle,math
    import numpy as np
    from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
    from PIL import Image,ImageOps    
    import numpy.ma as ma
    from PIL import Image
      
    img=plt.imread(panorama_fn)       
    print('\nseg shape={}'.format(img.shape))
    
    # define a grid matching the map size, subsample along with pixels
    theta=np.linspace(0, np.pi, img.shape[0])
    phi=np.linspace(0, 2*np.pi, img.shape[1])        
    print("theta shape={};phi shape={}".format(theta.shape,phi.shape))    
    theta,phi=np.meshgrid(theta, phi)
    theta=theta.T
    phi=phi.T
       
    print("theta shape={};phi shape={}".format(theta.shape,phi.shape))
    theta_phi=np.dstack((theta,phi))
    
    
    if FOV==True:
        verticalFOV_limit_ofVisual_field=[50,90-(-70)]
        horizontalFOV_visual_limit_field=[62,90-(-62)]
        horizontal_offset=0
        
        verticalFOV_limit_ofVisual_field_radians=[math.radians(d) for d in verticalFOV_limit_ofVisual_field]
        horizontalFOV_visual_limit_field_radians=[math.radians(d) for d in horizontalFOV_visual_limit_field]
        horizontal_offset_radians=math.radians(horizontal_offset)
        print(verticalFOV_limit_ofVisual_field_radians,horizontalFOV_visual_limit_field_radians,horizontal_offset_radians)
        
        mask=np.bitwise_and(theta>=verticalFOV_limit_ofVisual_field_radians[0], theta<=verticalFOV_limit_ofVisual_field_radians[1])
        theta=theta[mask]
        phi=phi[mask]
        img=img[mask]               

    R=50
    # sphere
    x=R * np.sin(theta) * np.cos(phi)
    y=R * np.sin(theta) * np.sin(phi)
    z=R * np.cos(theta)                
    print("x,y,z shape={},{},{}".format(x.shape,y.shape,z.shape))       

    # print(img)
    fig=mlab.figure(size=(600, 600),bgcolor=(1, 1, 1))  
    mlab.points3d(x, y, z, img/255,scale_factor=.25) #opacity=0.75,scale_factor=0.1     
    mlab.points3d(0, 0, 0,scale_factor=3,color=(1,0,0))
    
    # Plot the equator and the tropiques
    theta_equator=np.linspace(0, 2 * np.pi, 100)
    veiw_scope_dic={}
    for i,angle in enumerate([-math.radians(70), 0, math.radians(50)]):
        x_equator=R * np.cos(theta_equator) * np.cos(angle)
        y_equator=R * np.sin(theta_equator) * np.cos(angle)
        z_equator=R * np.ones_like(theta_equator) * np.sin(angle)    
        mlab.plot3d(x_equator, y_equator, z_equator, color=(0, 0, 0),opacity=0.6, tube_radius=None)  
        veiw_scope_dic[i]=[x_equator,y_equator,z_equator]
    
    str_info={0:'lower limit of visual filed:-70',1:'Standard line of sight:0',2:'Upper limit of visual filed:+50'}
    for k,v in str_info.items():
        mlab.text(veiw_scope_dic[k][0][0], veiw_scope_dic[k][1][0], v, z=veiw_scope_dic[k][2][0],width=0.025 * len(v), name=v,color=(0,0,0))

    
    vertical_label_radians=np.linspace(0, np.pi,14)
    vertical_label_degree=["{:.2f}".format(90-math.degrees(radi)) for radi in vertical_label_radians]
    phi_label=0
    for idx in range(len(vertical_label_radians)):
        theta_labe=vertical_label_radians[idx]
        x_label=R * np.sin(theta_labe) * np.cos(phi_label)
        y_label=R * np.sin(theta_labe) * np.sin(phi_label)
        z_label=R * np.cos(theta_labe)         
        mlab.points3d(x_label, y_label, z_label,scale_factor=1,color=(0,0,0))
        label=vertical_label_degree[idx]
        mlab.text(x_label, y_label, label, z=z_label,width=0.02 * len(label), name=label,color=(0,0,0))
        
    mlab.show()   
    
def array_classifier(array,n_classes=9):
    import mapclassify as mc
    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn import preprocessing
    
    array_shape=array.shape
    array_flatten=array.flatten()
    classifier=mc.NaturalBreaks(array_flatten,k=n_classes)
    print(classifier)
    
    classifications=pd.DataFrame(array).apply(classifier)
    classifications_array=classifications.to_numpy().reshape(array_shape)
    
    min_max_scaler=preprocessing.MinMaxScaler()
    classifications_array_standardization=min_max_scaler.fit_transform(classifications_array)
    classifications_object_change=Image.fromarray(np.uint8(classifications_array_standardization * 255) , 'L')
    classifications_object_change.save('./processed data/classifications_object_change.jpg')    
    
    return classifications_array

def auto_sphere_label(image_file):
    import math
    
    # create a figure window (and scene)
    fig = mlab.figure(size=(600, 600),bgcolor=(1, 1, 1))

    # load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    # print(texture)
    # (interpolate for a less raster appearance when zoomed in)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 50
    Nrad = 180

    # create the sphere source with a given radius and angular resolution
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,phi_resolution=Nrad)

    # print(sphere)
    # assemble rest of the pipeline, assign texture    
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)
    
    # Plot the equator and the tropiques
    theta_equator=np.linspace(0, 2 * np.pi, 100)
    veiw_scope_dic={}
    for i,angle in enumerate([-math.radians(70), 0, math.radians(50)]):
        x_equator=R * np.cos(theta_equator) * np.cos(angle)
        y_equator=R * np.sin(theta_equator) * np.cos(angle)
        z_equator=R * np.ones_like(theta_equator) * np.sin(angle)    
        mlab.plot3d(x_equator, y_equator, z_equator, color=(0, 0, 0),opacity=0.6, tube_radius=None)  
        veiw_scope_dic[i]=[x_equator,y_equator,z_equator]    
    
    str_info={0:'lower limit of visual filed:-70',1:'Standard line of sight:0',2:'Upper limit of visual filed:+50'}
    for k,v in str_info.items():
        mlab.text(veiw_scope_dic[k][0][0], veiw_scope_dic[k][1][0], v, z=veiw_scope_dic[k][2][0],width=0.029 * len(v), name=v,color=(0,0,0))
    
    vertical_label_radians=np.linspace(0, np.pi,14)
    vertical_label_degree=["{:.2f}".format(90-math.degrees(radi)) for radi in vertical_label_radians]
    phi_label=0
    for idx in range(len(vertical_label_radians)):
        theta_labe=vertical_label_radians[idx]
        x_label=R * np.sin(theta_labe) * np.cos(phi_label)
        y_label=R * np.sin(theta_labe) * np.sin(phi_label)
        z_label=R * np.cos(theta_labe)         
        mlab.points3d(x_label, y_label, z_label,scale_factor=1,color=(0,0,0))
        label=vertical_label_degree[idx]
        mlab.text(x_label, y_label, label, z=z_label,width=0.028 * len(label), name=label,color=(0,0,0))    
        
    # mlab.savefig('./processed data/img_sphere/1.jpg',size=(50,50))
    mlab.show()
    
def array_classifier_label(array,n_classes=9):
    import mapclassify as mc
    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn import preprocessing
    
    from PIL import Image,ImageOps
    import matplotlib.pyplot as plt
    import cv2
    import matplotlib
    import matplotlib.cm as cm
    
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
 
    pano_height,pano_width,=array.shape
    print(pano_width,pano_height)    
    
    # fig=plt.figure(figsize=(20,10))
    fig, ax=plt.subplots(figsize=(30,15))
    
    array_shape=array.shape
    array_flatten=array.flatten()
    classifier=mc.NaturalBreaks(array_flatten,k=n_classes)
    print(classifier)
    
    classifications=pd.DataFrame(array).apply(classifier)
    classifications_array=classifications.to_numpy().reshape(array_shape)
    
    min_max_scaler=preprocessing.MinMaxScaler()
    classifications_array_standardization=min_max_scaler.fit_transform(classifications_array)
    
    ax.imshow(np.uint8(classifications_array_standardization * 255),) #cmap=cm.gray
    ax.set_yticks(np.linspace(0,pano_height,len(vertical_label_degree)))    
    ax.set_yticklabels(vertical_label_degree) # provide name to the x axis tick marks   
    ax.set_xticks(np.linspace(0,pano_width,len(horizontal_label_degree)))    
    ax.set_xticklabels(horizontal_label_degree) # provide name to the x axis tick marks   
    ax.axhline(y=pano_height/2,color='r',linestyle='-.',linewidth=1)
    ax.axvline(x=pano_width/2,color='r',linestyle='-.',linewidth=1)
    # ax.set_title("Equirectangular format", va = 'bottom',fontsize=title_fontsize)  
 
    # plt.show()
    plt.legend()
    plt.savefig('./graph/classifications_object_change.jpg')
    
    
if __name__ == "__main__":
    #01 
    # image_file = './processed data/img_seg/27095509ca4819726949f3f0_45.jpg'
    # auto_sphere(image_file)
    # manual_sphere(image_file)
    # mlab.show()
    
    # mpl_sphere(image_file)
    
    #02
    # label_seg_path=r'./processed data/tourline_label_seg'
    # label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    # spherical_segs_pts_show(label_seg_fns[10],label_color)
    
    #03
    # img_object_change,pixels_diff_array_standardization=panorama_object_change(label_seg_path,label_color)
    # auto_sphere('./processed data/img_object_change.jpg')
    # spherical_img_pts_show('./processed data/img_object_change.jpg',FOV=False,)
    
    #04
    with open('./processed data/pixels_diff_array_standardization.pkl','rb') as f:
        pixels_diff_array_standardization=pickle.load(f)    
    
    
    # array_classifier_label(pixels_diff_array_standardization,n_classes=5)
    # spherical_img_pts_show('./processed data/classifications_object_change.jpg',FOV=False,)
    
    # classifications_array=array_classifier(pixels_diff_array_standardization,n_classes=5)
    spherical_img_pts_show('./processed data/classifications_object_change.jpg',FOV=False,)
    
    #05
    # spherical_segs_object_changing(label_seg_path,label_color)
    
    # fn=r'1e32241a3f2fe5a796ec847f_12.jpg'
    # img_panorama_fn=os.path.join('./data/panoramic imgs valid/',fn) #  
    # auto_sphere_label(img_panorama_fn)    
  
    # auto_sphere_label(os.path.join('./processed data/img_seg_redefined_color',fn))
    