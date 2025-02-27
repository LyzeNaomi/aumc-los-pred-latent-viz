
import torch
import numpy as np
import torch.nn as nn

from sklearn import metrics

class MSLELoss(nn.Module):
  def __init__(self):
      super().__init__()
      self.mse = nn.MSELoss()    
  def forward(self, true, pred):
      return self.mse(torch.log(true + 1), torch.log(pred + 1))
  
class Metrics():
  def diff_metrics(true, pred):
    mae = nn.L1Loss(reduction='sum')(pred, true)
    mse = nn.MSELoss(reduction='sum')(pred, true)
    log_true = torch.log1p(true) ; log_pred = torch.log1p(pred)
    msle = nn.MSELoss(reduction='sum')(log_pred, log_true)

    return [mae, mse, msle]