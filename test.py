import torch
import math
import torch.nn as nn
import torch.nn.functional as F

d_model=14
max_len=10

print("position:",torch.arange(0, max_len).unsqueeze(1).float())
position=torch.arange(0, max_len).unsqueeze(1).float()
print(torch.arange(0, d_model, 2))
# print(-(math.log(10000.0) / d_model))
# print(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
print("div_term:",10000.0**((torch.arange(0, d_model, 2).float())/d_model))
div_term = 10000.0**((torch.arange(0, d_model, 2).float())/d_model)
# print("pe:",torch.zeros(max_len, d_model))
# pe=torch.zeros(max_len, d_model)
# print(torch.sin(position * div_term))
# pe[:, 0::2]=torch.sin(position * div_term)
# print(pe)
#
# tensor_3d = torch.ones(2, 5, 4)
# print(tensor_3d)
# test_3d=torch.ones(2, 1, 4)*0.5
# print(test_3d)
# print(tensor_3d+test_3d[0:2])