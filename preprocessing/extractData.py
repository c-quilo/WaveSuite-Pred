import meshio
import pyvista as pv
import numpy as np
from os import listdir
from os.path import isfile, join

# Transform .vtk to .vtu

# Get list of files in directory

import meshio
import os
file_list = os.listdir("/path/to/vtk/")
print(file_list)
sortedList = sorted(file_list)
sortedList = sortedList[1:]
j = 0
for name in sortedList:
    mesh = meshio.read('/path/to/vtk' + name)
    mesh.write('/path/to/vtu/regularWave_' + str(j) + '.vtu')
    print(j)
    j = j + 1

#Save them in a folder 
file_list_vtu = os.listdir("/path/to/vtu/")

fieldname = 'nut'
collect_data_npy = []
for file_vtu in file_list_vtu:
    mesh = pv.read(file_vtu)
    collect_data_npy.append(mesh[fieldname])
collect_data_npy = np.array(collect_data_npy)
np.save(f"/path/to/vtu/regularWave_{fieldname}")