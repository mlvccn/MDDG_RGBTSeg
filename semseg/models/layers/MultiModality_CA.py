# Multi-Modality Cross Attention
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math
from natten.functional import na2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import sys
sys.path.append('/data/tangchen/MMSFormer')
from timm.models.layers import to_2tuple

class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
    
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
   
class ConvModule(nn.Module):
    def __init__(self, c1, c2, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = c1, 
            out_channels = c2, 
            kernel_size = kernel_size, 
            stride = 1, 
            groups = c1,
            padding = (kernel_size - 1) // 2, 
            bias = False
            )
        self.norm = nn.LayerNorm(c2)        # use SyncBN in original
        self.activate = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.norm(self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()))

class MLP(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
def generate_topk_cosine_mask(x, y, k_ration):
    """
    根据输入特征图 x 和 y 生成 Top-K 掩码。
    x, y: 形状为 (b, c, w, h) 的特征图。
    k: int, 表示取Top-K个最大余弦相似度的位置。
    
    返回值:
        mask: 形状为 (b, w, h) 的掩码矩阵,其中Top-K个位置为1,其他位置为0。
    """
    # print(x.shape)
    b, c, w, h = x.size()
    k = math.ceil(w * h * k_ration)
    # 将特征图重塑为 (b, c, N)，其中 N = w * h
    x_flat = x.view(b, c, -1)  # (b, c, N)
    y_flat = y.view(b, c, -1)  # (b, c, N)
    # 计算每个位置上的余弦相似度
    x_norm = F.normalize(x_flat, dim=1)  # (b, c, N)
    y_norm = F.normalize(y_flat, dim=1)  # (b, c, N)

    # output = b,N
    cosine_similarity = torch.sum(torch.mul(x_norm, y_norm),dim = 1)  # (b, N, N)，N=w*h

    # 对每个位置找到Top-K最小相似度的索引
    topk_values, topk_indices = torch.topk(-cosine_similarity, k, dim=-1)

    return topk_indices

class CML_Attiention_Mask(nn.Module):
    def __init__(self,dim:int, head:int,sr_ratio:int = 1,kernel_size:int = 3,dilation:int = 1):
        super().__init__()
        # natten.use_fused_na(True)
        # natten.use_autotuner(forward_pass = True, backward_pass= True)
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.rgb_q = nn.Linear(dim*2, dim)
        self.rgb_kv = nn.Linear(dim, dim*2)
        self.th_q = nn.Linear(dim*2, dim)
        self.th_kv = nn.Linear(dim, dim*2)
        # self.mid_proj = nn.Linear(2*dim, dim)
        self.rgb_proj = nn.Linear(dim, dim)
        self.th_proj = nn.Linear(dim, dim)
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.rgb_norm1 = nn.LayerNorm(dim)
        self.th_norm1 = nn.LayerNorm(dim)
        self.rgb_norm2 = nn.LayerNorm(dim)
        self.th_norm2 = nn.LayerNorm(dim)
        
        self.drop_path = DropPath(0.1)
        
        self.rgb_mlp = MLP(dim,2)
        self.th_mlp = MLP(dim,2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),  
        )
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
    def forward(self, input_rgb: Tensor, input_thermal: Tensor)-> Tensor:
        "rgb:b*c*w*h"
        input_rgb = input_rgb.permute(0, 2, 3, 1).contiguous()
        input_thermal = input_thermal.permute(0, 2, 3, 1).contiguous()
        
        rgb = self.rgb_norm1(input_rgb)
        thermal = self.th_norm1(input_thermal)
        "rgb:b*w*h*c"
        B, H, W, C = rgb.shape
        mid_feature = torch.cat([rgb, thermal], dim=-1).reshape(B, -1, 2*C) # ->b*(w*h)*2c
        gates = self.gate_mlp(mid_feature).reshape(B, H, W, 2)
        g_rgb, g_th = torch.sigmoid(gates[..., 0:1]), torch.sigmoid(gates[..., 1:2])  # [B, H, W, 1]
        rgb_q = self.rgb_q(mid_feature).reshape(B, H, W, self.head, C // self.head)
        th_q = self.th_q(mid_feature).reshape(B, H, W, self.head, C // self.head)
        rgb_k, rgb_v = self.rgb_kv(rgb).reshape(B, H, W, 2, self.head, C // self.head).permute(3, 0, 1, 2, 4, 5).contiguous()
        th_k, th_v = self.th_kv(thermal).reshape(B, H, W, 2, self.head, C // self.head).permute(3, 0, 1, 2, 4, 5).contiguous()
        # print(rgb_q.shape[0])
        # print(rgb_q.dtype)
        attn_rgb = na2d(rgb_q, -th_k, th_v, self.kernel_size, self.dilation, self.scale).reshape(B, H, W, C)
        attn_th = na2d(th_q, -rgb_k, rgb_v, self.kernel_size, self.dilation, self.scale).reshape(B, H, W, C)
        #fuse_rgb:B,H,W,C
        fuse_rgb = g_rgb * attn_rgb + (1 - g_rgb) * input_rgb
        fuse_th = g_th * attn_th + (1 - g_th) * input_thermal
        fuse_rgb = (input_rgb + self.drop_path(self.rgb_proj(fuse_rgb))).reshape(B, -1, C)
        fuse_th = (input_thermal + self.drop_path(self.th_proj(fuse_th))).reshape(B, -1, C)
        # fuse_rgb: B,N,C
        fuse_rgb = (fuse_rgb + self.drop_path(self.rgb_mlp(self.rgb_norm2(fuse_rgb), H, W))).reshape(B, H, W, C).permute(0,3,1,2).contiguous()
        fuse_th = (fuse_th + self.drop_path(self.th_mlp(self.th_norm2(fuse_th), H, W))).reshape(B, H, W, C).permute(0,3,1,2).contiguous()
        return fuse_rgb, fuse_th
                
class CML_Attiention(nn.Module):
    def __init__(self, dim: int, head: int, sr_ratio: int = 1, kernel_size: int = 3):
        super().__init__()
        self.sna_rgb = SparseNeighborhoodAttention(dim, kernel_size, head)
        self.sna_th = SparseNeighborhoodAttention(dim, kernel_size, head)

        self.rgb_norm1 = nn.LayerNorm(dim)
        self.th_norm1 = nn.LayerNorm(dim)
        self.rgb_norm2 = nn.LayerNorm(dim)
        self.th_norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(0.1)
        self.rgb_mlp = MLP(dim=dim, mlp_ratio=2)
        self.th_mlp = MLP(dim=dim, mlp_ratio=2)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def scatter_sparse_to_full(self, out_sparse, idx, N):
        B, num_q, C = out_sparse.shape
        out_full = torch.zeros(B, N, C, device=out_sparse.device, dtype=out_sparse.dtype)
        for b in range(B):
            out_full[b].index_copy_(0, idx[b], out_sparse[b])
        return out_full

    def forward(self, input_rgb: Tensor, input_thermal: Tensor, idx: Tensor) -> Tensor:
        input_rgb = input_rgb.permute(0, 2, 3, 1).contiguous()
        input_thermal = input_thermal.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = input_rgb.shape
        N = H * W
        for b in range(B):
            current_idx = idx[b]
            if current_idx.min() < 0 or current_idx.max() >= N:
                print(f"Error in batch {b}: index out of bounds, min={current_idx.min()}, max={current_idx.max()}, N={N}")
                raise RuntimeError("Index out of bounds")
        # sample sparse features
        input_rgb_flat = input_rgb.reshape(B, N, C)
        input_thermal_flat = input_thermal.reshape(B, N, C)
        # print(input_rgb_flat.shape)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)
        # print(idx_exp.shape)
        res_rgb = torch.gather(input_rgb_flat, dim=1, index=idx_exp).contiguous()
        res_thermal = torch.gather(input_thermal_flat, dim=1, index=idx_exp).contiguous()

        rgb = self.rgb_norm1(input_rgb).permute(0, 3, 1, 2).contiguous()
        thermal = self.th_norm1(input_thermal).permute(0, 3, 1, 2).contiguous()

        fuse_rgb = self.sna_rgb(thermal, rgb, idx, res_rgb)
        fuse_th = self.sna_th(rgb, thermal, idx, res_thermal) 

        fuse_rgb = fuse_rgb + self.drop_path(self.rgb_mlp(self.rgb_norm2(fuse_rgb)))
        fuse_th = fuse_th + self.drop_path(self.th_mlp(self.th_norm2(fuse_th)))

        rgb_full = self.scatter_sparse_to_full(fuse_rgb, idx, N)
        th_full = self.scatter_sparse_to_full(fuse_th, idx, N)

        rgb_full = rgb_full.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        th_full = th_full.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return rgb_full, th_full

class SparseNeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size=3, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim * 2, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpb = nn.Parameter(torch.zeros(num_heads, kernel_size * kernel_size))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Sigmoid()
        )

    def forward(self, x, y, idx, res_feat=None):
        B, C, H, W = x.shape
        N = H * W
        num_q = idx.shape[1]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        y_flat = y.permute(0, 2, 3, 1).reshape(B, N, C)
        y_cat = torch.cat([x_flat, y_flat], dim=-1)

        kv_all = self.kv(x_flat).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q_all = self.q(y_cat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_all, V_all = kv_all[0], kv_all[1]

        K_map = K_all.permute(0, 2, 1, 3).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        V_map = V_all.permute(0, 2, 1, 3).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        pad = self.kernel_size // 2
        unfold_K = F.unfold(F.pad(K_map, (pad, pad, pad, pad)), kernel_size=self.kernel_size)
        unfold_V = F.unfold(F.pad(V_map, (pad, pad, pad, pad)), kernel_size=self.kernel_size)

        unfold_K = unfold_K.transpose(1, 2).reshape(B, N, self.kernel_size**2, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        unfold_V = unfold_V.transpose(1, 2).reshape(B, N, self.kernel_size**2, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        out = torch.zeros(B, num_q, self.dim, device=x.device)
        for b in range(B):
            Q_b = Q_all[b, :, idx[b]]
            K_b = unfold_K[b, :, idx[b]]
            V_b = unfold_V[b, :, idx[b]]

            attn = (Q_b.unsqueeze(2) * -K_b).sum(-1) * self.scale
            attn = attn + self.rpb.unsqueeze(1)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            out_b = torch.einsum("hqk,hqkd->hqd", attn, V_b)
            out_b = out_b.transpose(0, 1).reshape(num_q, self.dim)
            out_b = self.proj_drop(self.proj(out_b))  # [num_q, dim]

            # gate fusion with residual
            res_b = res_feat[b]  # [num_q, dim]
            gate_input = y_cat[b, idx[b]]  # [num_q, 2C]
            gate = self.gate_mlp(gate_input)  # [num_q, 2]
            gate_attn = gate[:, 0].unsqueeze(-1)  # [num_q, 1]
            gate_res = gate[:, 1].unsqueeze(-1)
            out_b = gate_attn * out_b + gate_res * res_b
            out[b] = out_b
            
        return out  # [B, num_q, C]
    
def test_sparse_neighborhood_attention_on_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.set_device(0)
    
    B, C, H, W = 2, 64, 8, 8
    num_heads = 8
    kernel_size = 3

    # 初始化模块
    sna = CML_Attiention(dim=C, kernel_size=kernel_size, head=num_heads).to(device)

    # 随机输入图像并启用梯度
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    y = torch.randn(B, C, H, W, device=device, requires_grad=True)
    idx = generate_topk_cosine_mask(x, y, 0.5)
   
    # 前向传播
    rgb, th = sna(x, y, idx)
    print(rgb.shape)
    print(idx)
    print(f"Input shape: {x.shape}")
    print(f"Query idx shape: {idx.shape}")
    print(f"Output shape: {rgb.shape}")

    assert rgb.shape == (B, C, H, W), "输出形状不正确！"

    # 检查反向传播
    loss = rgb.mean()
    loss.backward()

    # 检查梯度是否正常
    assert x.grad is not None, "输入没有梯度，反向传播失败！"
    print("SparseNeighborhoodAttention Passed on GPU!")

# 运行测试
test_sparse_neighborhood_attention_on_gpu()
