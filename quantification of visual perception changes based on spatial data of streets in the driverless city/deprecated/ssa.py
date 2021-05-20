import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mpl_sphere(image_file):
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
    
    
#已知四点经纬度位置计算球面面积

if __name__ == "__main__":
    image_file = './data/sample/v2.jfif'
    mpl_sphere(image_file)
    plt.show()