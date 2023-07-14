import torch
#import torch.nn as nn 
#import torch.nn.functional as F 
#from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
#from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import Sequential, GCNConv 



#class taken from https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/12
class View(torch.nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def __repr__(self):
        return f'View{self.columns}'

    def forward(self, input):
        '''
        Reshapes the input according to the number of columns required saved in the view data structure.
        '''
        out = input.view(-1, self.columns)
        return out
    


# In[ ]:  Model1-variant1

class classic_AE_variant1(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim):
        
        # Init parent
        super(classic_AE_variant1, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                ( View(num_nodes) , 'x -> x') ,
            
                                #AE_encoder
                                ( Linear(num_nodes, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes) , 'x -> x'), 
            
                                #The output activation function of the model goes here, but in this case it is None 
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
# In[ ]:  Model1-variant3


class classic_AE_variant3(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim):
        
        # Init parent
        super(classic_AE_variant3, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                ( View(num_nodes) , 'x -> x') ,
            
                                #AE_encoder
                                ( Linear(num_nodes, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes) , 'x -> x'), 
            
                                #This entry is the output activation function of the model 
                                ( Sigmoid() , 'x -> x'),  
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x


    
# In[ ]:  Model2-variant1

class GCN_AE_variant1(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim, embedding_sequence):
        
        # Init parent
        super(GCN_AE_variant1, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                #GCN layers
                                (GCNConv(1, embedding_sequence[0]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[0] , embedding_sequence[1]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[1] , embedding_sequence[2]), 'x, edge_index -> x'),
            
                                ( View(num_nodes*embedding_sequence[2]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
            
                                #AE_encoder
                                ( Linear(num_nodes*embedding_sequence[2], 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes) , 'x -> x'), 
            
                                #The output activation function of the model goes here, but in this case it is None 
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
    
# In[ ]:  Model2-variant3

class GCN_AE_variant3(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim, embedding_sequence):
        
        # Init parent
        super(GCN_AE_variant3, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                #GCN layers
                                (GCNConv(1, embedding_sequence[0]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[0] , embedding_sequence[1]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[1] , embedding_sequence[2]), 'x, edge_index -> x'),
            
                                ( View(num_nodes*embedding_sequence[2]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
            
                                #AE_encoder
                                ( Linear(num_nodes*embedding_sequence[2], 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes) , 'x -> x'), 
            
                                #This entry is the output activation function of the model 
                                ( Sigmoid() , 'x -> x'),  
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
    
# In[ ]:  Model3-variant1


class GCN_AE_GCN_variant1(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim, embedding_sequence):
        
        # Init parent
        super(GCN_AE_GCN_variant1, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                #GCN layers
                                (GCNConv(1, embedding_sequence[0]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[0] , embedding_sequence[1]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[1] , embedding_sequence[2]), 'x, edge_index -> x'),
            
                                ( View(num_nodes*embedding_sequence[2]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
            
                                #AE_encoder
                                ( Linear(num_nodes*embedding_sequence[2], 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes*embedding_sequence[3]) , 'x -> x'), 
            
                                ( View(embedding_sequence[3]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
                                
                                #GCN layers
                                (GCNConv(embedding_sequence[3] , embedding_sequence[4] ), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[4] , embedding_sequence[5] ), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[5] , 1 ), 'x, edge_index -> x'),
            
                                ( View(num_nodes) , 'x -> x') ,
            
                                #The output activation function of the model goes here, but in this case it is None 
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
    
# In[ ]:  Model3-variant2


class GCN_AE_GCN_variant2(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim, embedding_sequence):
        
        # Init parent
        super(GCN_AE_GCN_variant2, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                #GCN layers
                                (GCNConv(1, embedding_sequence[0]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[0] , embedding_sequence[1]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[1] , embedding_sequence[2]), 'x, edge_index -> x'),
            
                                ( View(num_nodes*embedding_sequence[2]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
            
                                #AE_encoder
                                ( Linear(num_nodes*embedding_sequence[2], 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes*embedding_sequence[3]) , 'x -> x'), 
            
                                ( View(embedding_sequence[3]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
                                
                                #GCN layers
                                (GCNConv(embedding_sequence[3] , embedding_sequence[4] ), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[4] , embedding_sequence[5] ), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[5] , 1 ), 'x, edge_index -> x'),
            
                                ( View(num_nodes) , 'x -> x') ,
            
                                #This entry is the output activation function of the model 
                                ( ReLU() , 'x -> x'),  
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
    
# In[ ]:  Model3-variant3

class GCN_AE_GCN_variant3(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim, embedding_sequence):
        
        # Init parent
        super(GCN_AE_GCN_variant3, self).__init__() 
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                #GCN layers
                                (GCNConv(1, embedding_sequence[0]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[0] , embedding_sequence[1]), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[1] , embedding_sequence[2]), 'x, edge_index -> x'),
            
                                ( View(num_nodes*embedding_sequence[2]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
            
                                #AE_encoder
                                ( Linear(num_nodes*embedding_sequence[2], 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes*embedding_sequence[3]) , 'x -> x'), 
            
                                ( View(embedding_sequence[3]) , 'x -> x') ,
                                # Another activation function would go here, but in this case it is None
                                
                                #GCN layers
                                (GCNConv(embedding_sequence[3] , embedding_sequence[4] ), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[4] , embedding_sequence[5] ), 'x, edge_index -> x'),
                                ( ReLU() , 'x -> x'),
                                (GCNConv(embedding_sequence[5] , 1 ), 'x, edge_index -> x'),
            
                                ( View(num_nodes) , 'x -> x') ,
            
                                #This entry is the output activation function of the model 
                                ( Sigmoid() , 'x -> x'),  
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
    
# NON-CANON


# In[ ]:  Model1-variant2

class classic_AE_variant2(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim):
        
        # Init parent
        super(classic_AE_variant2, self).__init__() 
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                ( View(num_nodes) , 'x -> x') ,
            
                                #AE_encoder
                                ( Linear(num_nodes, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, num_nodes) , 'x -> x'), 
            
                                #This entry is the output activation function of the model 
                                ( ReLU() , 'x -> x'),  
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x
    
    
    
# In[ ]:  Extra layer model

class classic_AE_variant1_extra_layer(torch.nn.Module ):
    def __init__(self, num_nodes, latent_space_dim):
        
        # Init parent
        super(classic_AE_variant1_extra_layer, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = Sequential('x, edge_index', [
                                ( View(num_nodes) , 'x -> x') ,
            
                                #AE_encoder
                                ( Linear(num_nodes, 4*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(4*latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, latent_space_dim) , 'x -> x'),
        ])
        
        self.decoder = Sequential('x, edge_index', [
                                #AE_decoder
                                ( Linear(latent_space_dim, 2*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(2*latent_space_dim, 4*latent_space_dim) , 'x -> x'),
                                ( ReLU() , 'x -> x'),
                                ( Linear(4*latent_space_dim, num_nodes) , 'x -> x'), 
            
                                #The output activation function of the model goes here, but in this case it is None 
        ])
        
    def forward(self, x, edge_index):  
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x 