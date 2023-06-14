import numpy as np
import pyvista as pv
from os import listdir
from os.path import isfile, join

def gridInterpolation(mesh, dimx, dimy, dimz):
    # Find bounds of the original mesh
    bounds = mesh.bounds
    # Create a target grid of dimensions 'x'x'y'x'z' that spans the entire extent of the input mesh.
    x = np.linspace(bounds[0], bounds[1], dimx)
    y = np.linspace(bounds[2], bounds[3], dimy)
    z = np.linspace(bounds[4], bounds[5], dimz)
    x, y, z = np.meshgrid(x, y, z)
    
    grid = pv.StructuredGrid(x, y, z)

    # Now you can interpolate from the input mesh to the target grid.
    interpolated_mesh = grid.interpolate(mesh)
    return interpolated_mesh


# PATH = './irregularWave/'
# FIELD = 'U'
# onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
# onlyfiles = np.sort(onlyfiles)
# n = len(onlyfiles)
# dimx = 32
# dimy = 32
# dimz = 32
# collect_nut = np.zeros((n, dimx, dimy, dimz))
# collect_U = np.zeros((n, dimx, dimy, dimz, 3))

# j = 0
# for file in onlyfiles:
#     filename = f'{PATH}{file}'
#     print(filename)
#     mesh = pv.read(filename)
#     temp = gridInterpolation(mesh, dimx, dimy, dimz)
#     collect_U[j, :, :, :, :] = np.reshape(temp['U'], (dimx, dimy, dimz, 3), order='F')
#     collect_nut[j, :, :, :] = np.reshape(temp['nut'], (dimx, dimy, dimz), order='F')

#     j = j+1

# np.save(f'./interpolatedGrids/griddedIW_U.npy', collect_U)
# np.save(f'./interpolatedGrids/griddedIW_nut.npy', collect_nut)

PATH = './regularWave/'
FIELD = 'U'
onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
onlyfiles = np.sort(onlyfiles)
n = len(onlyfiles)
dimx = 32
dimy = 32
dimz = 32
collect_nut = np.zeros((n, dimx, dimy, dimz))
collect_U = np.zeros((n, dimx, dimy, dimz, 3))

j = 0
for file in onlyfiles:
    filename = f'{PATH}{file}'
    print(filename)
    mesh = pv.read(filename)
    temp = gridInterpolation(mesh, dimx, dimy, dimz)
    collect_U[j, :, :, :, :] = np.reshape(temp['U'], (dimx, dimy, dimz, 3), order='F')
    collect_nut[j, :, :, :] = np.reshape(temp['nut'], (dimx, dimy, dimz), order='F')

    j = j+1

np.save(f'./interpolatedGrids/griddedRW_U.npy', collect_U)
np.save(f'./interpolatedGrids/griddedRW_nut.npy', collect_nut)





