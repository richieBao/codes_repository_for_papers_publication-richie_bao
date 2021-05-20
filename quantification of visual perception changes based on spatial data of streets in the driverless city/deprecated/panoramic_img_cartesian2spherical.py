# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:37:32 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

def cartesian2spherical_coordinates(img_fn):    
    img=Image.open(img_fn)
    img_array=np.asarray(img)
    
    #img resolution
    img_w_px=1024
    img_h_px=512
    
    #camera field of view angles
    img_ha_deg=359 #fov=[10,360]
    img_va_deg=89 
    
    #camera rotaton angles
    hcam_deg=0 #heading [0,360] baidu
    vcam_deg=0 #pitch [0,90] baidu
    
    #camera rotation angles in radians
    hcam_rad=hcam_deg/180.0*math.pi
    vcam_rad=vcam_deg/180.0*math.pi
    
    #Rotation around y-axis for vertical rotation of camera
    rot_y=np.matrix([
        [math.cos(vcam_rad), 0, math.sin(vcam_rad)],
        [0, 1, 0],
        [-math.sin(vcam_rad), 0, math.cos(vcam_rad)]
    ])
    print(rot_y)
    rot_z=np.matrix([
        [math.cos(hcam_rad), -math.sin(hcam_rad), 0],
        [math.sin(hcam_rad), math.cos(hcam_rad), 0],
        [0, 0, 1]
    ])    
    print(rot_z)
    
    geo_coordi=[]
    spherical_coordi=[]
    polar=[]
    for x in tqdm(range(1,img_h_px)):
        for y in range(1,img_w_px):
            # print(i,j)
            
            #calculate relative position to center in degrees
            # p_theta=(j - img_w_px / 2.0) / img_w_px * img_ha_deg / 180.0 * math.pi
            # p_phi = -(i - img_h_px / 2.0) / img_h_px * img_va_deg / 180.0 * math.pi
            
            #transform into cartesian coordinates
            # p_x = math.cos(p_phi) * math.cos(p_theta)
            # p_y = math.cos(p_phi) * math.sin(p_theta)
            # p_z = math.sin(p_phi)
            # p0 =np.matrix( [[p_x], [p_y], [p_z]])
            
            #Apply rotation matrices (note, z-axis is the vertical one)
            #first vertically
            # p1=rot_y * p0;
            # p2=rot_z * p1;            
            # print(p2)
            
            #Transform back into spherical coordinates
            # theta=math.atan2(p2[1], p2[0])
            # phi=math.asin(p2[2])     
            
            # theta=math.atan2(p2[1], p2[0])
            # phi=math.asin(p2[2])
            
            #Retrieve longitude,latitude
            # longitude=theta / math.pi * 180.0;
            # latitude=phi / math.pi * 180.0;            
            
            # print(longitude,latitude)
            # geo_coordi.append((longitude,latitude))
            
            
            # Converting cartesian to polar coordinate
            # Calculating radius
            radius = math.sqrt( x * x + y * y )          
            # Calculating angle (theta) in radian
            theta = math.atan(y/x)            
            # Converting theta from radian to degree
            theta = 180 * theta/math.pi            
                                    
            polar.append((radius,theta))
            
            #Retrieve longitude,latitude
            # longitude=theta / math.pi * 180.0
            # latitude=radius/ math.pi * 180.0          
            # geo_coordi.append((longitude,latitude))
            
        #     break
        # break
    # return geo_coordi
    return polar,geo_coordi
    




if __name__=="__main__":
    img_fn='./data/sample/v2.jfif'
    a,b=cartesian2spherical_coordinates(img_fn)
