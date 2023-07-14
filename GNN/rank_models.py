"""Functions to rank and plot the losses results from the tested models inside a jupyter notebook"""


import math
import pandas as pd
import matplotlib.pyplot as plt 



def plot_losses(results_variants, variants_names, filename=None, figsize=(16,12)): 
    """Plot the (ideally) decaying loss of the models during the epochs.
    References:   I used a little help from this links: 
    # https://stackoverflow.com/questions/25239933/how-to-add-title-to-subplots-in-matplotlib
    # https://stackoverflow.com/questions/27016904/matplotlib-legends-in-subplot

    Parameters
    ----------

    results_variants: dictionary
    Each entry stores the losses of different models. The keys are the models' names and each value is a list of dictionaries describing the errors. 
    
    variants_names: list of strings
    Names of the models that want to be read from the results_variants dictionary as keys.    

    filename: string
    If a file location is inputted instead of None, then the figure is saved in that location. Default is None.  

    figsize:tuple
    Size of the figure. Default is (16,12). 
    """ 
    
    
    columns = int(math.ceil(len(variants_names)/2))
    
    if figsize is None:
        figsize = (16, 4*columns)

    fig, axs = plt.subplots(2, columns , figsize=figsize) 

    axs = axs.reshape(-1)

    for i, var_name in enumerate(variants_names):
        current_result = results_variants[var_name]

        train_loss = [res['log loss'] for res in current_result]
        val_loss = [res['val_log loss'] for res in current_result]

        axs[i].plot( train_loss , label='train loss')
        axs[i].plot( val_loss , label='val loss' )
        axs[i].title.set_text(var_name) 
        axs[i].set_xlabel('epoch')
        axs[i].legend()

    if filename is not None:
        fig.savefig('saved_figures/' + filename)
    
    
    
def rank_losses(results_variants, variants_names, selected_epoch = 20, 
                first_column_model_name=None, filename=None):
    """Plot the (ideally) decaying loss of the models during the epochs

    Parameters
    ----------

    results_variants: dictionary
    Each entry stores the losses of different models. The keys are the models' names and each value is a list of dictionaries describing the errors. 
    
    variants_names: list of strings
    Names of the models that want to be read from the results_variants dictionary as keys.    

    selected_epoch:int
    Epoch number in which the results should be compared. 

    first_column_model_name:string
    If a string is inputted instead of None, then an extra column is added containing that string. Default is None.

    filename: string
    If a string indicating file location is inputted instead of None, then the figure is saved in that location. Default is None.  
    """ 
    
    final_losses = []

    for i, var_name in enumerate(variants_names):
        current_result = results_variants[var_name]
        val_loss_selected_epoch = current_result[selected_epoch-1]['val_log loss']
        final_losses.append(val_loss_selected_epoch)

    variants_and_losses = pd.DataFrame.from_dict({'variants_name':variants_names, 'final_validation_loss':final_losses})
    variants_and_losses = variants_and_losses.sort_values(by='final_validation_loss').reset_index(drop=True)
    
    if first_column_model_name:
        df_columns = list(variants_and_losses.columns)
        
        variants_and_losses['Model name'] = first_column_model_name # 'Model Number I'
        variants_and_losses = variants_and_losses[['Model name']+df_columns]
        
    if filename:
        variants_and_losses.to_csv('saved_csv/'+filename, index=False )
        print('CSV was saved')  
    
    return variants_and_losses