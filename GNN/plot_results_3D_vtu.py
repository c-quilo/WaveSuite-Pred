#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Parts of this code were taken and modified from the following sources: https://github.com/DL-WG/ROMS-tutorial/blob/main/MNIST_AE.ipynb 


import numpy as np
import matplotlib.pyplot as plt 
import pyvista as pv

import torch

from torch_geometric.loader import DataLoader
from train_top_AE import evaluate_top_AE

import random as random_package


# In[ ]: Compare three models using point variable 'nut'


def compare_three_plots_models_3D_vtu(ae_model1, ae_model2, ae_model3, 
                                graph_data_list_input, nDisplay, filename_to_copy_structure, 
                                      feature_select='nut',  device=torch.device("cuda"), 
                                show_edges=False, random=True, index_sequence=[1, 15, 30] , 
                                      set_pred_labels_as = ["Model1 Predictions", "Model2 Predictions", "Model3 Predictions"], 
                                batch_size=32, save_fig=False, filename='no_name.eps'):
    
    #Get the random indices to plot
    if random:
        ##This line is different because we need to get different numbers in the sample
        # randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
        randomIndex = np.array( random_package.sample(range( len(graph_data_list_input) ), nDisplay ) )
    else:
        randomIndex = index_sequence[: min(len(graph_data_list_input),len(index_sequence)) ]
        
    print('Indexes plotted', randomIndex) 
    nDisplay = len(randomIndex)
    
    #Create data loader but this time do notshuffle it
    graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
    data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
    # Get the results when you test the model
    y_pred1, y_real1 = evaluate_top_AE(ae_model1, data_loader, device) 
    y_pred2, y_real2 = evaluate_top_AE(ae_model2, data_loader, device)
    y_pred3, y_real3 = evaluate_top_AE(ae_model3, data_loader, device)
    
    # NOW DO THE PLOTTING 
    
    # Read one file with the fixed structure 
    template = pv.read(filename_to_copy_structure)
    template.rename_array(old_name='nut', new_name='nut_cell', preference='cell')
    mesh = template.copy() 
    
    #feature_select = 'TracerBackground'
    mesh.set_active_scalars(feature_select)
    
    #Create the plot mesh
    pl = pv.Plotter(shape=(4, nDisplay))
    
    for i in range(nDisplay):
        
        pl.subplot(0, i)
        mesh = template.copy()
        mesh[feature_select] = y_real1[i]
        mesh.set_active_scalars(feature_select)
        
        single_slice = mesh.slice(normal=[0, 1, 0])
        pl.add_mesh(single_slice, clim=[0, 0.32] , cmap='plasma')
        pl.camera_position = 'xz'
        pl.camera.zoom(1.7)
        pl.add_text('Real', font_size=10)
        
        pl.subplot(1, i)
        mesh = template.copy()
        mesh[feature_select] = y_pred1[i]
        mesh.set_active_scalars(feature_select)
        
        single_slice = mesh.slice(normal=[0, 1, 0])
        pl.add_mesh(single_slice, clim=[0, 0.32] , cmap='plasma')  #, show_edges=show_edges, reset_camera=True
        pl.camera_position = 'xz'
        pl.camera.zoom(1.7)
        pl.add_text(set_pred_labels_as[0], font_size=10) #"Model1 Predictions"
        
        pl.subplot(2, i)
        mesh = template.copy()
        mesh[feature_select] = y_pred2[i]
        mesh.set_active_scalars(feature_select)
        
        single_slice = mesh.slice(normal=[0, 1, 0])
        pl.add_mesh(single_slice, clim=[0, 0.32] , cmap='plasma')  #, show_edges=show_edges, reset_camera=True
        pl.camera_position = 'xz'
        pl.camera.zoom(1.7)
        pl.add_text(set_pred_labels_as[1], font_size=10) #"Model2 Predictions"
        
        pl.subplot(3, i)
        mesh = template.copy()
        mesh[feature_select] = y_pred3[i]
        mesh.set_active_scalars(feature_select)
        
        single_slice = mesh.slice(normal=[0, 1, 0])
        pl.add_mesh(single_slice, clim=[0, 0.32] , cmap='plasma')  #, show_edges=show_edges, reset_camera=True
        pl.camera_position = 'xz'
        pl.camera.zoom(1.7)
        pl.add_text(set_pred_labels_as[2], font_size=10) #"Model3 Predictions"        
        
    if save_fig:
        pl.save_graphic('saved_figures/' + filename)  
    
    pl.show()
    
    
    
    
def plot_model_3D_vtu(ae_model, graph_data_list_input, nDisplay, filename_to_copy_structure, 
                      device=torch.device("cuda"), 
                      show_edges=False, random=True, index_sequence=[1, 2, 3] , is_data_in_loader=False, 
                      batch_size=32, save_fig=False, filename='no_name.eps'): 
    
    if is_data_in_loader:
        data_loader = graph_data_list_input
        
        # Get the results when you test the model
        y_pred, y_real = evaluate(ae_model, data_loader)
    
        #Get the random indices to plot
        if random:
            #randomIndex = np.random.randint(0, y_pred.shape[0], nDisplay)
            randomIndex = np.array( random_package.sample(range( y_pred.shape[0] ), nDisplay ) )
        else:
            randomIndex = index_sequence[: min(y_pred.shape[0],len(index_sequence)) ]
            
        print('Indexes plotted', randomIndex) 
        nDisplay = len(randomIndex)
        indices = randomIndex
        
    else:
        
        #Get the random indices to plot
        if random: 
            #randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
            randomIndex = np.array( random_package.sample(range( len(graph_data_list_input) ), nDisplay ) )
        else:
            randomIndex = index_sequence[:len(graph_data_list_input)]
            
        print('Indexes plotted', randomIndex) 
        nDisplay = len(randomIndex)
        
        #Create data loader but this time DO NOT shuffle it
        graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
        data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
        # Get the results when you test the model
        y_pred, y_real = evaluate_top_AE(ae_model, data_loader, device) 
        
        indices = range(nDisplay)
        
    # NOW DO THE PLOTTING 
    
    # Read one file with the fixed structure 
    template = pv.read(filename_to_copy_structure)
    template.rename_array(old_name='nut', new_name='nut_cell', preference='cell')
    mesh = template.copy() 
    
    feature_select = 'nut'
    mesh.set_active_scalars(feature_select)
    
    #Create the plot mesh
    pl = pv.Plotter(shape=(2, nDisplay))
    
    for i, j in enumerate(indices):
        
        pl.subplot(0, i)
        mesh = template.copy()
        mesh[feature_select] = y_real[j]
        mesh.set_active_scalars(feature_select)        
        
        single_slice = mesh.slice(normal=[0, 1, 0])
        pl.add_mesh(single_slice, clim=[0, 0.32] , cmap='plasma')  #, show_edges=show_edges, reset_camera=True
        pl.camera_position = 'xz'
        pl.camera.zoom(1.7)  
        pl.add_text("Real Image", font_size=15)
        
        pl.subplot(1, i)
        mesh = template.copy()
        mesh[feature_select] = y_pred[j]
        mesh.set_active_scalars(feature_select)
        
        single_slice = mesh.slice(normal=[0, 1, 0])
        pl.add_mesh(single_slice, clim=[0, 0.32] , cmap='plasma')  #, show_edges=show_edges, reset_camera=True
        pl.camera_position = 'xz'
        pl.camera.zoom(1.7)  
        pl.add_text("Model Predictions", font_size=15)
    
    if save_fig:  #https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.save_graphic.html 
        pl.save_graphic('saved_figures/' + filename)  
    
    pl.show()
    
    