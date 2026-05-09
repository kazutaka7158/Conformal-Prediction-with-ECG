import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        '''
        input x: (B,C,N)
        target y: (B,N)
        CE_loss -log(y_pred): (B,N)
        y_pred: (B,N)
        '''
        CE_loss = F.cross_entropy(input, target, reduction='none')
        y_pred = torch.exp(-CE_loss)
        Focal_loss = torch.mean((1 - y_pred) ** self.gamma * CE_loss)

        return Focal_loss

def dice_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(0, 2)) 
    total = pred.sum(dim=(0, 2)) + target.sum(dim=(0, 2))

    dice = (2. * intersection + smooth) / (total + smooth)

    return dice