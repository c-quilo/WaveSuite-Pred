#!/usr/bin/env python
# coding: utf-8

# For these functions, I reused code from the Machine Learning module of the 
# MSc "Environmental Data Science and Machine Learning" and modified it accordingly. 


# In[ ]:

import numpy as np

import torch
import torch.nn as nn 

from AE_top_models import classic_AE_variant1, classic_AE_variant3, classic_AE_variant1_extra_layer, \
                            GCN_AE_variant1, GCN_AE_variant3,  \
                            GCN_AE_GCN_variant1, GCN_AE_GCN_variant2, GCN_AE_GCN_variant3

#SHould I? 
from AE_top_models import classic_AE_variant2

from livelossplot import PlotLosses 

import random


## Use GPU for training
## Inside the functions the default is:  
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
## However, for very large datasets it is recommended to do:  device = torch.device("cpu")  given that despite taking longer times to run it has more memory. 


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True


# For training


# In[ ]:


def train_top_AE_epoch(ae_model, optimizer, criterion, data_loader, num_nodes,
                   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") , seed=42):
    
    set_seed(seed)
    
    ## Set the model to mode TRAIN
    ae_model.train()
    
    batch_size = data_loader.batch_size
    
    train_loss = 0
    
    for batch in data_loader:
        
        batch = batch.to(device)  # or is it only batch.to(device) ?
        
        ## Reset gradients
        optimizer.zero_grad()
        
        ## Compute the current output values of the model...
        batch_train = batch.x.float()
        temp_train_decoded = ae_model(batch_train, batch.edge_index) 
        # ... and make them the same shape
        batch_train = batch_train.view(temp_train_decoded.shape[0],temp_train_decoded.shape[1])
        
        ## Compute the loss ...
        ## (Remember that label in this case is the whole batch because the model is an AE)
        # loss = criterion(temp_train_decoded, batch_train ) 
        loss = ((temp_train_decoded - batch_train)**2).mean()
        
        # ... and the gradients of the parameters ...
        loss.backward()
        ## .. to update the parameters using the gradients
        optimizer.step()  
        
        ## ... And finally add the loss in every batch (and scale the loss according to every batch of the epoch)
        train_loss += loss*(batch.size(0)/num_nodes)  
                
    train_loss = train_loss/len(data_loader.dataset)
    return train_loss


# In[ ]:


def validate_top_AE_epoch(ae_model, criterion, data_loader, num_nodes ,
                      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") , seed=42):
    
    set_seed(seed)
    
    ## Set the model to mode EVAL
    ae_model.eval()
    
    batch_size = data_loader.batch_size
    
    validation_loss = 0 
    
    for batch in data_loader:
        
        # torch.no_grad() saves time by telling the model not to take the gradients into account, 
        # because the model has already been trained
        with torch.no_grad(): 
            batch = batch.to(device) # or is it only batch.to(device) ?
            
            ## Compute the current output values of the model...
            batch_validation = batch.x.float()
            temp_validation_decoded = ae_model(batch_validation, batch.edge_index) #batch_index = batch.batch
            # ... and make them the same shape
            batch_validation = batch_validation.view(temp_validation_decoded.shape[0],temp_validation_decoded.shape[1])
            
            ## Compute the loss ...
            ## (Remember that label in this case is the whole batch because the model is an AE)
            # loss = criterion(temp_validation_decoded, batch_validation ) 
            loss = ((temp_validation_decoded - batch_validation)**2).mean()
            
            ## ... and add the loss in every batch (and scale the loss according to every batch of the epoch)
            validation_loss += loss*(batch.size(0)/num_nodes)  
                
    validation_loss = validation_loss/len(data_loader.dataset)
    return validation_loss


# In[ ]:


def train_top_AE(train_loader, validation_loader, 
             num_nodes, num_features, embedding_sequence, latent_space_dim,
             ae_model_type='classic_AE_variant1', num_epochs=50, plot_losses=True, 
             seed=42, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ,
             use_sgd_instead_of_adam=False, lr=1e-1, momentum=0.2):
    
    set_seed(seed)
    
    invalid_entry = False
    
    
    if ae_model_type=='classic_AE_variant1':
        model = classic_AE_variant1(num_nodes, latent_space_dim).to(device) #This one does not have embedding sequence
    elif ae_model_type=='classic_AE_variant3':
        model = classic_AE_variant3(num_nodes, latent_space_dim).to(device) #This one does not have embedding sequence
        
    elif ae_model_type=='GCN_AE_variant1':
        model = GCN_AE_variant1(num_nodes, latent_space_dim, embedding_sequence).to(device) #This one does not have embedding sequence
    elif ae_model_type=='GCN_AE_variant3':
        model = GCN_AE_variant3(num_nodes, latent_space_dim, embedding_sequence).to(device) #This one does not have embedding sequence
    
    elif ae_model_type=='GCN_AE_GCN_variant1':
        model = GCN_AE_GCN_variant1(num_nodes, latent_space_dim, embedding_sequence).to(device) #This one does not have embedding sequence
    elif ae_model_type=='GCN_AE_GCN_variant2':
        model = GCN_AE_GCN_variant2(num_nodes, latent_space_dim, embedding_sequence).to(device) #This one does not have embedding sequence
    elif ae_model_type=='GCN_AE_GCN_variant3':
        model = GCN_AE_GCN_variant3(num_nodes, latent_space_dim, embedding_sequence).to(device) #This one does not have embedding sequence  
    
    
    elif ae_model_type=='classic_AE_variant2':
        model = classic_AE_variant2(num_nodes, latent_space_dim ).to(device) #This one does not have embedding sequence  
    elif ae_model_type=='classic_AE_variant1_extra_layer':
        model = classic_AE_variant1_extra_layer(num_nodes, latent_space_dim ).to(device) #This one does not have embedding sequence  
    
    
    else:
        invalid_entry = True
        print('\n An invalid ae_model_type was provided, a classic AE was used as default')
        model = classic_AE_variant3(num_nodes, latent_space_dim).to(device)
    
    ## The ADAM is the default, since the ADAM optimizer performs better in this case compared against SGD  
    optimizer = torch.optim.Adam(model.parameters())
    if use_sgd_instead_of_adam:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    ## The loss here is the mean_squared_error because the model is an AE
    criterion = nn.MSELoss()
    
    if plot_losses:
        liveloss = PlotLosses()
    loss_register = []
    
    for epoch in range(num_epochs):
        #During each epoch update the parameters
        train_temp_loss = train_top_AE_epoch(model, optimizer, criterion, train_loader, num_nodes, device)
        
        #Optional: Check the validation loss in each epoch
        validation_temp_loss = validate_top_AE_epoch(model, criterion, validation_loader, num_nodes, device) 
        
        #Compute the losses in each iteration 
        logs = {}
        logs['' + 'log loss'] = train_temp_loss.item()
        logs['val_' + 'log loss'] = validation_temp_loss.item()
        loss_register.append(logs)
        
        #Optionally plot the train and validation loss   
        if plot_losses:
            liveloss.update(logs)
            liveloss.draw()
            
    if invalid_entry == True:
        print('\n An invalid ae_model_type was provided, a classic AE was used as default') 
        
    print("\n Number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    #At the end just return 1. the model with the trained parameters and 2. the list of losses in every epoch
    return model, loss_register 


# In[ ]:


# For evaluating after training


# In[ ]:


def evaluate_top_AE(ae_model, data_loader, 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ):
    
    ae_model.eval()
    
    batch_size = data_loader.batch_size  
    
    ys, y_preds = [], []
    
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.to(device)  # or is it only batch.to(device) ?
            
            ## Compute the current output values of the model
            batch_test = batch.x.float()
            temp_test_decoded = ae_model(batch_test, batch.edge_index) 
            
            #Identify y and y_pred
            y = batch_test
            y_pred = temp_test_decoded
            
            #Reshape y to have the same shape as y_pred
            y = y.reshape(y_pred.shape)
            
            #Append the results to each list 
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            
    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0) 




def model_MSE_error(ae_model, data_loader, error_type='mean'):
    
    # Get the results when you test the model
    y_pred, y_real = evaluate(ae_model, data_loader)
    
    if error_type=='mean':
        error = ((y_pred-y_real)**2).mean() 
    elif error_type=='sum':
        error = ((y_pred-y_real)**2).sum() 
    else:
        error = ((y_pred-y_real)**2).mean() 
        print('Invalid error_type was provided, mean was used as default')
    
    return error 