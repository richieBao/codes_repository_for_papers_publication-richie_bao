import cv2

import numpy as np

img = cv2.imread('./data/sample/v2.jfif')
rows = img.shape[0]
cols = img.shape[1]
blank = np.zeros_like(img)
#圆心定为图片中心
center_x = int(rows / 2)
center_y = int(cols / 2)
#假设球的半径
r = int(((rows**2+cols**2)**0.5)/2)+20
#假设映射平面位于 z = r 处
pz = r
for x in range(rows):
    ox = x
    x = x - center_x
    for y in range(cols):
        oy = y    
        y = y - center_y
        z = (r*r - x*x - y*y)**0.5
        
        #假设观看映射点为(0,0,2r)
        k = (pz - 2*r)/(z - 2*r)
        px = int(k*x)
        py = int(k*y)
        px = px + center_x
        py = py + center_y
        blank[px , py, :] = img[ox , oy ,:]
cv2.imwrite('out.jpg',blank)
