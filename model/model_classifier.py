import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)

        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        # [B, L1, D1], [B, L2, D2]
        B, L1, _ = x1.size()
        L2 = x2.size(1)

        q1 = self.proj_q1(x1).view(B, L1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)   # [B, H, L1, d_k]
        k2 = self.proj_k2(x2).view(B, L2, self.num_heads, self.k_dim).permute(0, 2, 3, 1) # [B, H, d_k, L2]
        v2 = self.proj_v2(x2).view(B, L2, self.num_heads, self.v_dim).permute(0, 2, 1, 3) # [B, H, L2, d_v]

        attn = torch.matmul(q1, k2) / math.sqrt(self.k_dim)  # [B, H, L1, L2]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = torch.matmul(attn, v2)  # [B, H, L1, d_v]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L1, -1)  # [B, L1, H*d_v]

        out = self.proj_o(out)

        return out, attn

class CrossAttentionClassifier(nn.Module):
    def __init__(self, feature_model, high_dim_size=1024, low_dim_size=13, num_classes=10, hidden_size=256,
                 contrast=False):
        super(CrossAttentionClassifier, self).__init__()

        self.feature_model = feature_model
        self.hidden_size = hidden_size
        self.debug = False
        self.contrast = contrast

        # 高维特征处理
        self.high_dim_processor = nn.Sequential(
            nn.Linear(high_dim_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01),
        )

        # 低维特征处理
        self.low_dim_processor = nn.Sequential(
            nn.Linear(low_dim_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01),
        )

        # 交叉注意力机制
        self.cross_attention = CrossAttention(
            in_dim1=hidden_size,  # image feature
            in_dim2=hidden_size,  # key feature
            k_dim=hidden_size // 2,
            v_dim=hidden_size // 2,
            num_heads=2
        )

        # 柔性残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """  """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # 小偏置避免死神经元

    def forward(self, x, low_x):
        # image feature
        _, high_x = self.feature_model(x)  # (B, high_dim_size)

        # image feature mapping
        high_processed = self.high_dim_processor(high_x)  # (B, hidden_size)
        high_unsqueezed = high_processed.unsqueeze(1)  # (B, 1, hidden_size)

        # low feature process
        low_processed = self.low_dim_processor(low_x)  # (B, hidden_size)


        low_unsqueezed = low_processed.unsqueeze(1)  # (B, 1, hidden_size)

        attn_output, attn_weights = self.cross_attention(
            low_unsqueezed,
            high_unsqueezed,
        )  # (B, 1, hidden_size)

        output = self.classifier(attn_output.squeeze(1))

        if self.contrast:
            idx = torch.randperm(low_unsqueezed.size(0))
            x_b_cf = low_unsqueezed[idx]
            attn_output_contrast, attn_weights_contrast = self.cross_attention(
                x_b_cf,
                high_unsqueezed,
            )

            attn_output_X = F.normalize(attn_output.squeeze(1), dim=-1)  # (B, 256)
            attn_output_contrast = F.normalize(attn_output_contrast.squeeze(1), dim=-1)

            logits = torch.matmul(attn_output_X, attn_output_contrast.T) / 1.0

            labels = torch.arange(logits.size(0), device=logits.device)

            loss_contrast = F.cross_entropy(logits, labels)

            return output, loss_contrast

        return output, attn_weights

def get_classifier(feature_model, high_dim_size=1024, low_dim_size=13, num_classes=3):
    return CrossAttentionClassifier(feature_model, high_dim_size, low_dim_size, num_classes)