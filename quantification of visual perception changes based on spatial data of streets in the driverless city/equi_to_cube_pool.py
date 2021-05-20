# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:44:29 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
Reference and Transfer: https://github.com/bingsyslab/360projection
"""
import cv2
import numpy as np
from PIL import Image

def deg2rad(d):
  return float(d) * np.pi / 180

def rotate_image(old_image,channel_num=1):
    if channel_num==3:
        (old_height, old_width, _) = old_image.shape
    elif channel_num==1:
        (old_height, old_width, ) = old_image.shape
    M = cv2.getRotationMatrix2D(((old_width - 1) / 2., (old_height - 1) / 2.), 270, 1)
    rotated = cv2.warpAffine(old_image, M, (old_width, old_height))
    return rotated

def xrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def yrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render_image_np(theta0, phi0, fov_h, fov_v, width, img,channel_num=1):
    """
    theta0 is pitch
    phi0 is yaw
    render view at (pitch, yaw) with fov_h by fov_v
    width is the number of horizontal pixels in the view
    """
    m = np.dot(yrotation(phi0), xrotation(theta0))
    
    if channel_num==3:
        (base_height, base_width, _) = img.shape
    elif channel_num==1:
        (base_height, base_width, ) = img.shape
    
    
    height = int(width * np.tan(fov_v / 2) / np.tan(fov_h / 2))
    
    if channel_num==3:
        new_img = np.zeros((height, width, 3), np.uint8)
    elif channel_num==1:
        new_img = np.zeros((height, width, ), np.uint8)
    
      
    DI = np.ones((height * width, 3), np.int)
    trans = np.array([[2.*np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                      [0., -2.*np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])
    
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    DI[:, 0] = xx.reshape(height * width)
    DI[:, 1] = yy.reshape(height * width)
    
    v = np.ones((height * width, 3), np.float)
    
    v[:, :2] = np.dot(DI, trans.T)
    v = np.dot(v, m.T)
    
    diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
    theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
    phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi
    
    ey = np.rint(theta * base_height / np.pi).astype(np.int)
    ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)
    
    ex[ex >= base_width] = base_width - 1
    ey[ey >= base_height] = base_height - 1  
    
    # print(ey.shape,ex.shape)
    new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
    return new_img

  
def equi_to_cube(face_size, img,channel_num=1):        
      """
      given an equirectangular spherical image, project it onto standard cube
      """
      cube_img_h = face_size * 3
      cube_img_w = face_size * 2
      if channel_num==3:
          cube_img = np.zeros((cube_img_h, cube_img_w, 3), np.uint8)
      elif channel_num==1:
         cube_img = np.zeros((cube_img_h, cube_img_w, ), np.uint8)
    
      ii = render_image_np(np.pi / 2, np.pi, \
              np.pi / 2, np.pi / 2, \
              face_size, img,channel_num=channel_num)
    #   cv2.imwrite('g_top.jpg', ii)

      cube_img[:int(cube_img_h / 3), int(cube_img_w / 2):] = ii.copy()
    
      ii = render_image_np(0, 0, \
              np.pi / 2, np.pi / 2, \
              face_size, img,channel_num=channel_num)
    #   cv2.imwrite('g_front.jpg', ii)
    
      cube_img[int(cube_img_h / 3):int(cube_img_h * 2 / 3), :int(cube_img_w / 2)] = rotate_image(ii,channel_num=channel_num).copy()
    
      ii = render_image_np(0, np.pi / 2, \
              np.pi / 2, np.pi / 2, \
              face_size, img,channel_num=channel_num)
    #   cv2.imwrite('g_right.jpg', ii)
    
      cube_img[int(cube_img_h * 2 / 3):, :int(cube_img_w / 2)] = rotate_image(ii,channel_num=channel_num).copy()
    
      ii = render_image_np(0, np.pi, \
              np.pi / 2, np.pi / 2, \
              face_size, img,channel_num=channel_num)
    #   cv2.imwrite('g_back.jpg', ii)
    
      cube_img[int(cube_img_h / 3):int(cube_img_h * 2 / 3), int(cube_img_w / 2):] = ii.copy()
    
      ii = render_image_np(0, np.pi * 3 / 2, \
              np.pi / 2, np.pi / 2, \
              face_size, img,channel_num=channel_num)
    #   cv2.imwrite('g_left.jpg', ii)
    
      cube_img[:int(cube_img_h / 3), :int(cube_img_w / 2)] = rotate_image(ii,channel_num=channel_num).copy()
    
      ii = render_image_np(-np.pi / 2, np.pi, \
              np.pi / 2, np.pi / 2, \
              face_size, img,channel_num=channel_num)
    #   cv2.imwrite('g_bottom.jpg', ii)
    
      cube_img[int(cube_img_h * 2 / 3):, int(cube_img_w / 2):] = ii.copy()
    
    #   cv2.imwrite('g_cube.jpg', cube_img)
      return cube_img

def labels_equi2cube(label_seg_path,face_size):
    import glob,os,pickle
    from tqdm import tqdm
    from pathlib import Path
    import numpy as np
    
    label_seg_fns=glob.glob(os.path.join(label_seg_path,'*.pkl'))
    # print(label_seg_fns)
    for label_seg_fn in tqdm(label_seg_fns):
        with open(label_seg_fn,'rb') as f:
            label_seg=pickle.load(f)#.cpu().detach().numpy() 
        fn_stem=Path(label_seg_fn).stem
        fn_key,fn_idx=fn_stem.split("_")
        
        label_seg_cube=equi_to_cube(face_size,label_seg)
        with open(os.path.join('./processed data/label_seg_cube','{}.pkl'.format(fn_stem)),'wb') as f:
            pickle.dump(label_seg_cube,f)        
        
        label_seg=label_seg.cpu().detach().numpy()
        label_seg_panorama=np.array([label_color[v] for v in label_seg.flatten()]).reshape((label_seg.shape[0],label_seg.shape[1],3))
        label_panorama=Image.fromarray(label_seg_panorama.astype(np.uint8))   
        label_panorama.save(os.path.join('./processed data/img_seg_redefined_color','{}.jpg'.format(fn_stem)))        
        
        label_seg_cube_color=np.array([label_color[v] for v in label_seg_cube.flatten()]).reshape((label_seg_cube.shape[0],label_seg_cube.shape[1],3))
        label_cube=Image.fromarray(label_seg_cube_color.astype(np.uint8))  
        label_cube=label_cube.rotate(90, expand=1)
        label_cube.save(os.path.join('./processed data/img_seg_cube','{}.jpg'.format(fn_stem)))
        
        # break

face_size=1000    
def labels_equi2cube_pool(label_seg_fn): #face_size
    import glob,os,pickle
    from tqdm import tqdm
    from pathlib import Path
    import numpy as np
    
    with open(label_seg_fn,'rb') as f:
        label_seg=pickle.load(f)#.cpu().detach().numpy() 
    fn_stem=Path(label_seg_fn).stem
    fn_key,fn_idx=fn_stem.split("_")
    
    label_seg_cube=equi_to_cube(face_size,label_seg)
    with open(os.path.join('./processed data/tourline_label_seg_cube','{}.pkl'.format(fn_stem)),'wb') as f: #'./processed data/label_seg_cube'
        pickle.dump(label_seg_cube,f)        
    
    label_seg=label_seg.cpu().detach().numpy()
    label_seg_panorama=np.array([label_color[v] for v in label_seg.flatten()]).reshape((label_seg.shape[0],label_seg.shape[1],3))
    label_panorama=Image.fromarray(label_seg_panorama.astype(np.uint8))   
    label_panorama.save(os.path.join('./processed data/tourline_img_seg_redefined_color','{}.jpg'.format(fn_stem)))  #'./processed data/img_seg_redefined_color'      
    
    label_seg_cube_color=np.array([label_color[v] for v in label_seg_cube.flatten()]).reshape((label_seg_cube.shape[0],label_seg_cube.shape[1],3))
    label_cube=Image.fromarray(label_seg_cube_color.astype(np.uint8))  
    label_cube=label_cube.rotate(90, expand=1)
    label_cube.save(os.path.join('./processed data/tourline_img_seg_cube','{}.jpg'.format(fn_stem)))    #'./processed data/img_seg_cube'
    
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

def imgs__equi2cube(pano_path,face_size):
    import glob,os
    from tqdm import tqdm
    from pathlib import Path
    
    pano_path_fns=glob.glob(os.path.join(pano_path,'*.jpg'))
    # print(pano_path_fns)
    for fn in tqdm(pano_path_fns):
        fn_stem=Path(fn).stem
        fn_key,fn_idx=fn_stem.split("_")
        
        img=cv2.imread(fn)
        cube_img=equi_to_cube(face_size,img,channel_num=3)
        cube_img=np.rot90(cube_img)
        cv2.imwrite(os.path.join('./processed data/img_cube','{}.jpg'.format(fn_stem)), cube_img) 

        # break

if __name__ == '__main__':    
    label_seg_path=r'./processed data/label_seg'
    face_size=1000
    # labels_equi2cube(label_seg_path,face_size)

    # img=cv2.imread('./data/panoramic imgs valid/00f786b9f3703f4b9f9d683a_0.jpg')
    # cube_img=equi_to_cube(face_size, img)
    # cv2.imwrite('./processed data/temp/cub_image_%d_%d.bmp' % (pitch, yaw), cube_img)    
    pano_path='./data/panoramic imgs valid'
    imgs__equi2cube(pano_path,face_size)
    
    
    # yaw = 0
    # pitch = 0    
    # fov_h = 90
    # fov_v = 90    
    # rimg = render_image_np(deg2rad(pitch), deg2rad(yaw), \
    #                     deg2rad(fov_v), deg2rad(fov_h), \
    #                     face_size, img)
    # cv2.imwrite('./processed data/temp/rendered_image_%d_%d.bmp' % (pitch, yaw), rimg)
