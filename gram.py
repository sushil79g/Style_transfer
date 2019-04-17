import torch
import numpy as np
def gram_matrix(array_tensor):
  ar_size = array_tensor.size()
  depth, height, width = ar_size[1],ar_size[2],ar_size[3]
  
  tensor = array_tensor.view(depth, height*width)
  return (torch.mm(tensor, tensor.t()))