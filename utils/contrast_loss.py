import torch
import torch.nn.functional as F

import numpy as np
import torch.fft as fft

def cosine_similarity_loss(x1, x2, x3, margin=1):
    
    
    pos_cosine = torch.sum(x1 * x3, dim=-1) / (torch.norm(x1, dim=-1) * torch.norm(x3, dim=-1))
    
    neg_cosine_1 = torch.sum(x1 * x2, dim=-1) / (torch.norm(x1, dim=-1) * torch.norm(x2, dim=-1))  
    neg_cosine_2 = torch.sum(x3 * x2, dim=-1) / (torch.norm(x3, dim=-1) * torch.norm(x2, dim=-1))  

    loss = torch.mean(torch.clamp(neg_cosine_1 + neg_cosine_2 - pos_cosine + margin, min=0.0))
    
    return loss


def mask_matrix(x, device="cuda:0"):
    padded_tensor = F.pad(x, (0, 0, 0, 432),"constant", 0).to(device) 
    mask = torch.ones(16, 512).to(device)  
    mask[:, 160:] = 0  
    mask = mask.unsqueeze(1).unsqueeze(2)
    # mask = mask.bool()

    return padded_tensor,mask







