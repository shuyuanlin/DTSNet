import torch
import torch.nn as nn
from torch.nn import functional as F

# 余弦相似度损失
def cosine_similarity_loss(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))
    return loss
