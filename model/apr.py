import torch
import torch.nn as nn
import torch.nn.functional as F

class APR(nn.Module):
    def __init__(self, base=64, temp=0.5, sigma=0.1):
        super(APR, self).__init__()

        self.temp = temp
        self.sigma = sigma

        self.recs = nn.ModuleList([
            self._make_rec(base * 4, base * 8),
            self._make_rec(base * 8, base * 16),
            self._make_rec(base * 16, base * 32)
        ])

    def _make_rec(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        )

    def get_attention(self, preds, temp):
        """
        计算空间注意力掩膜
        preds: [Bs, C, W, H]
        返回: [Bs, H, W]
        """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        fea_map = value.mean(dim=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        return S_attention

    def forward(self, f_s, f_t):

        f_rec = []
        for k in range(len(self.recs)):
            f_s_k = f_s[k]
            f_t_k = f_t[k]

            A_T = self.get_attention(f_t_k, self.temp)

            std_fs = torch.std(f_s_k, dim=1, keepdim=True)
            sigma = self.sigma * std_fs * A_T.unsqueeze(1)
            noise = torch.randn_like(f_s_k) * sigma

            f_s_noisy = f_s_k + noise
            f_r_k = self.recs[k](f_s_noisy)
            f_rec.append(f_r_k)

        return f_rec
