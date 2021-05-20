from mayavi import mlab
from tvtk.api import tvtk # python wrappers for the C++ vtk ecosystem


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


import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
import matplotlib.pyplot as plt # only for manipulating the input image

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
    print(texture)
    mesh.actor.actor.texture = texture

    # tell mayavi that the mapping from points to pixels happens via a sphere
    mesh.actor.tcoord_generator_mode = 'sphere' # map is already given for a spherical mapping
    cylinder_mapper = mesh.actor.tcoord_generator
    # caveat 2: if prevent_seam is 1 (default), half the image is used to map half the sphere
    cylinder_mapper.prevent_seam = 0 # use 360 degrees, might cause seam but no fake data
    #cylinder_mapper.center = np.array([0,0,0])  # set non-trivial center for the mapping sphere if necessary

if __name__ == "__main__":
    image_file = './data/sample/v2.jfif'
    auto_sphere(image_file)
    # manual_sphere(image_file)
    mlab.show()