import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from models.transformer import get_transformer_encoder

from utils.contrast_loss import mask_matrix

import torch.nn.functional as F


# Anomaly Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, linear_embedding, linear_embedding_contrast, transformer_encoder, mlp_layers, mlp_layers_contrast, d_embed, patch_size, max_seq_len,aug_window):
        """
        <class init args>
        linear_embedding : embedding layer to feed data into Transformer encoder
        transformer_encoder : Transformer encoder body
        mlp_layers : MLP layers to return output data
        d_embed : embedding dimension (in Transformer encoder)
        patch_size : number of data points for an embedded vector
        max_seq_len : maximum length of sequence (= window size)
        """
        super(AnomalyTransformer, self).__init__()
        self.linear_embedding = linear_embedding
        self.linear_embedding_contrast = linear_embedding_contrast
        self.transformer_encoder = transformer_encoder
        self.mlp_layers = mlp_layers
        self.mlp_layers_contrast = mlp_layers_contrast 

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.data_seq_len = patch_size * max_seq_len
        self.mask_matrix = mask_matrix #hj
        self.fc1 = nn.Linear(aug_window * 32, d_embed)
        self.fc2 = nn.Linear(d_embed, 64 * d_embed)
        self.d_embed = d_embed

    def forward(self, x,contrast=False):
        """
        <input info>
        x : (n_batch, n_token, d_data) = (_, max_seq_len*patch_size, _)
        """
        
        if contrast:
            n_batch = x.shape[0] 
            embedded_out = x 
            embedded_out = self.linear_embedding_contrast(embedded_out)
            embedded_out = embedded_out.transpose(1,2)
            embedded_out = nn.functional.interpolate(embedded_out, size=512, mode='linear', align_corners=True)
            embedded_out = embedded_out.transpose(1,2)
            # embedded_out = embedded_out.view(-1,self.d_embed,self.d_embed)
            mask =None
        else:
            n_batch = x.shape[0]
            embedded_out = x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
            embedded_out = self.linear_embedding(embedded_out)  
            mask = None
            
        transformer_out = self.transformer_encoder(embedded_out,mask)  

        if not contrast:
            output = self.mlp_layers(transformer_out)  
            return output.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.data_seq_len, -1) 
        else:
            return transformer_out 
    
    
    
# Get Anomaly Transformer.
def get_anomaly_transformer(input_d_data,
                            output_d_data,
                            patch_size,
                            d_embed=512,
                            hidden_dim_rate=4.,
                            max_seq_len=512,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            transformer_n_layer=12,
                            transformer_n_head=8,
                            dropout=0.1,aug_window=80):
    """
    <input info>
    input_d_data : data input dimension
    output_d_data : data output dimension
    patch_size : number of data points per embedded feature
    d_embed : embedding dimension (in Transformer encoder)
    hidden_dim_rate : hidden layer dimension rate to d_embed
    max_seq_len : maximum length of sequence (= window size)
    positional_encoding : positional encoding for embedded input; None/Sinusoidal/Absolute
    relative_position_embedding : relative position embedding option
    transformer_n_layer : number of Transformer encoder layers
    transformer_n_head : number of heads in multi-head attention module
    dropout : dropout rate
    """
    hidden_dim = int(hidden_dim_rate * d_embed)
    
    linear_embedding = nn.Linear(input_d_data*patch_size, d_embed)
    # linear_embedding_contrast = nn.Linear(input_d_data, d_embed)
    linear_embedding_contrast = nn.Linear(input_d_data, d_embed)

    transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  relative_position_embedding=relative_position_embedding,
                                                  n_layer=transformer_n_layer,
                                                  n_head=transformer_n_head,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout)
    mlp_layers = nn.Sequential(nn.Linear(d_embed, hidden_dim),
                               nn.GELU(),
                               nn.Linear(hidden_dim, output_d_data*patch_size))
    
    mlp_layers_contrast = nn.Sequential(nn.Linear(d_embed, hidden_dim), ##
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, 1))
    
    nn.init.xavier_uniform_(linear_embedding.weight)
    nn.init.zeros_(linear_embedding.bias)
    nn.init.xavier_uniform_(mlp_layers[0].weight)
    nn.init.zeros_(mlp_layers[0].bias)
    nn.init.xavier_uniform_(mlp_layers[2].weight)
    nn.init.zeros_(mlp_layers[2].bias)

    nn.init.xavier_uniform_(linear_embedding_contrast.weight)
    nn.init.zeros_(linear_embedding_contrast.bias)
    nn.init.xavier_uniform_(mlp_layers_contrast[0].weight)
    nn.init.zeros_(mlp_layers_contrast[0].bias)
    nn.init.xavier_uniform_(mlp_layers_contrast[2].weight)
    nn.init.zeros_(mlp_layers_contrast[2].bias)
    
    return AnomalyTransformer(linear_embedding,
                              linear_embedding_contrast,
                              transformer_encoder,
                              mlp_layers,
                              mlp_layers_contrast,
                              d_embed,
                              patch_size,
                              max_seq_len,aug_window=aug_window)


        
