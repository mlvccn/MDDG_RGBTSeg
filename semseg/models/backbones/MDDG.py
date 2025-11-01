import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import torch.nn.init as init
from semseg.models.layers import ChannelMambaFusionBlock,ChannelMambaFusionBlock_Mask
from semseg.models.layers import CML_Attiention, CML_Attiention_Mask
import math

def generate_topk_cosine_mask(x, y, k_ration):
    # print(x.shape)
    b, c, w, h = x.size()
    k = math.ceil(w * h * k_ration)
   
    x_flat = x.view(b, c, -1)  # (b, c, N)
    y_flat = y.view(b, c, -1)  # (b, c, N)
   
    x_norm = F.normalize(x_flat, dim=1)  # (b, c, N)
    y_norm = F.normalize(y_flat, dim=1)  # (b, c, N)

    # output = b,N
    cosine_similarity = torch.sum(torch.mul(x_norm, y_norm),dim = 1)  # (b, N, N)，N=w*h

    topk_values, topk_indices = torch.topk(-cosine_similarity, k, dim=-1)
    
    mask = torch.zeros_like(cosine_similarity)

    mask.scatter_(1, topk_indices, 1)
    mask = (mask - cosine_similarity).detach() + cosine_similarity
    raw_mask = -cosine_similarity.view(b, w, h).unsqueeze(dim=1)
    mask = mask.view(b, w, h).unsqueeze(dim=1)
    return mask, raw_mask, topk_indices

class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

    
class CustomDWConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CustomPWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))


class FusionBlock(nn.Module):
    def __init__(self, channels, reduction=8, num_modals=2, mask_impl = True):
        super(FusionBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction # 似乎未启用
        self.num_modals = num_modals
        self.mask_impl = mask_impl
        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])
        
        self.mix_ffn = nn.ModuleList([
            MixFFN(self.channels[0], self.channels[0]),
            MixFFN(self.channels[1], self.channels[1]),
            MixFFN(self.channels[2], self.channels[2]),
            MixFFN(self.channels[3], self.channels[3]),
        ])

        self.channel_attns = nn.ModuleList([
            ChannelAttentionBlock(self.channels[0]),
            ChannelAttentionBlock(self.channels[1]),
            ChannelAttentionBlock(self.channels[2]),
            ChannelAttentionBlock(self.channels[3]),
        ])
        # [64, 128, 320, 512]
        if mask_impl:
            self.Mamba_block = nn.ModuleList([
                ChannelMambaFusionBlock_Mask(hidden_dim = self.channels[0], head = 32, ssm_ratio = 4,mlp_ratio = 2),
                ChannelMambaFusionBlock_Mask(hidden_dim = self.channels[1], head = 32, ssm_ratio = 4,mlp_ratio = 2),
                ChannelMambaFusionBlock_Mask(hidden_dim = self.channels[2], head = 32, ssm_ratio = 2,mlp_ratio = 2),
                ChannelMambaFusionBlock_Mask(hidden_dim = self.channels[3], head = 32, ssm_ratio = 2,mlp_ratio = 2), # head = 64
                ])

            self.CML_block = nn.ModuleList([
                CML_Attiention_Mask(dim = self.channels[0], kernel_size = 13, head = 2),
                CML_Attiention_Mask(dim = self.channels[1], kernel_size = 7, head = 4),
                CML_Attiention_Mask(dim = self.channels[2], kernel_size = 5, head = 10),
                CML_Attiention_Mask(dim = self.channels[3], kernel_size = 5, head = 16),
            ])
        else:
            self.Mamba_block = nn.ModuleList([
                ChannelMambaFusionBlock(hidden_dim = self.channels[0], head = 32, ssm_ratio = 4,mlp_ratio = 2),
                ChannelMambaFusionBlock(hidden_dim = self.channels[1], head = 32, ssm_ratio = 4,mlp_ratio = 2),
                ChannelMambaFusionBlock(hidden_dim = self.channels[2], head = 32, ssm_ratio = 2,mlp_ratio = 2),
                ChannelMambaFusionBlock(hidden_dim = self.channels[3], head = 32, ssm_ratio = 2,mlp_ratio = 2), # head = 64
                ])

            self.CML_block = nn.ModuleList([
                CML_Attiention(dim = self.channels[0], kernel_size = 13, head = 2),
                CML_Attiention(dim = self.channels[1], kernel_size = 7, head = 4),
                CML_Attiention(dim = self.channels[2], kernel_size = 5, head = 10),
                CML_Attiention(dim = self.channels[3], kernel_size = 5, head = 16),
            ])
        self.norm_block = nn.ModuleList([
            nn.LayerNorm(2*channels[0]),
            nn.LayerNorm(2*channels[1]),
            nn.LayerNorm(2*channels[2]),
            nn.LayerNorm(2*channels[3]),
            ])
        # Initialize linear fusion layers with Kaiming initialization
        for linear_layer in self.liner_fusion_layers:
            init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        if self.mask_impl:
            mask ,raw_mask,_ = generate_topk_cosine_mask(x[0],x[1],0.5)
            x_rgb,x_e = self.CML_block[layer_idx](x[0],x[1])
            y_rgb,y_e  = self.Mamba_block[layer_idx](x_rgb,x_e) 
            # y_rgb,y_e = self.CML_block[layer_idx](x_rgb,x_e)
            y_rgb = mask * y_rgb + x[0]
            y_e = mask * y_e + x[1]
        else:
            mask ,raw_mask, idx= generate_topk_cosine_mask(x[0],x[1],0.5)
            x_rgb,x_e = self.CML_block[layer_idx](x[0],x[1],idx)
            y_rgb,y_e  = self.Mamba_block[layer_idx](x_rgb,x_e,idx) 
            
        x = torch.cat([y_rgb,y_e], dim = 1)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_block[layer_idx](x)
        x_sum = self.liner_fusion_layers[layer_idx](x)
        x_sum = self.mix_ffn[layer_idx](x_sum, H, W) + self.channel_attns[layer_idx](x_sum, H, W)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2), y_rgb, y_e, raw_mask


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        # shape B*N*H*D
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 可以尝试把这几行改换使用 F.scaled_dot_product_attention 
        # F.scaled_dot_product_attention 
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))
        
        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x 


mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class MixTransformer(nn.Module):
    def __init__(self, model_name: str = 'B0', modality: str = 'depth'):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        # self.model_name = 'B2'
        self.model_name = model_name
        # TODO: Must comment the following line later
        # self.model_name = 'B2' if modality == 'depth' else model_name
        embed_dims, depths = mit_settings[self.model_name] 
        self.modality = modality  
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
   
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Initialize with pretrained weights
        self.init_weights()

    def init_weights(self):
        print(f"Initializing weight for {self.modality}...")
        checkpoint = torch.load(f'checkpoints/pretrained/segformer/mit_{self.model_name.lower()}.pth', map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        msg = self.load_state_dict(checkpoint, strict=False)
        del checkpoint
        print(f"Weight init complete with message: {msg}")
     
    def forward(self, x: Tensor) -> list:
        x_cam = x       
        
        B = x_cam.shape[0]
        outs = []
        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x4_cam)

        return outs


class MDDG(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
   
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        
        # Have extra modality
        if self.num_modals > 0:
            # Backbones and Fusion Block for extra modalities
            self.extra_mit = nn.ModuleList([MixTransformer('B1', self.modals[i]) for i in range(self.num_modals)])
            self.fusion_block = FusionBlock(self.channels, reduction=16, num_modals=self.num_modals+1)
     
    def forward(self, x: list) -> list:
        x_cam = x[0]        
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]
        outs = []
        rgb_outs = []
        th_outs = []
        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed1(x_ext[i])
                for blk in self.extra_mit[i].block1:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm1(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_rgb, x_th , raw_mask= self.fusion_block([x1_cam, *x_ext], layer_idx=0)
            th_outs.append(*x_ext)
            x_ext[0] = x_th
            outs.append(x_fused)
            rgb_outs.append(x_rgb)
        else:
            outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x_rgb)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed2(x_ext[i])
                for blk in self.extra_mit[i].block2:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm2(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_rgb, x_th , _ = self.fusion_block([x2_cam, *x_ext], layer_idx=1)
            th_outs.append(*x_ext)
            x_ext[0] = x_th
            rgb_outs.append(x_rgb)
            outs.append(x_fused)
        else:
            outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x_rgb)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed3(x_ext[i])
                for blk in self.extra_mit[i].block3:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm3(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_rgb, x_th ,  _= self.fusion_block([x3_cam, *x_ext], layer_idx=2)
            th_outs.append(*x_ext)
            x_ext[0] = x_th
            outs.append(x_fused)
            rgb_outs.append(x_rgb)
        else:
            outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x_rgb)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed4(x_ext[i])
                for blk in self.extra_mit[i].block4:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm4(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_rgb, x_th , _ = self.fusion_block([x4_cam, *x_ext], layer_idx=3)
            th_outs.append(*x_ext)
            x_ext[0] = x_th
            rgb_outs.append(x_rgb)
            outs.append(x_fused)
            
        else:
            outs.append(x4_cam)
        
        return outs, rgb_outs, th_outs, raw_mask


if __name__ == '__main__':
    modals = ['img', 'aolp', 'dolp', 'nir']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = MDDG('B2', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)

