from pointnet2_utils import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        self.fc_delta = nn.Sequential(
            nn.Linear(k, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    def forward(self, xyz, features):

        xyz = xyz.permute(0,2,1)
        features = features.permute(0,2,1)

        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]

        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre

        res = res.permute(0,2,1)
        return res, attn

