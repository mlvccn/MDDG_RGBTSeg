import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F
from torch import fft
from timm.models.layers import to_2tuple
from functools import partial
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from fvcore.nn import flop_count_table, FlopCountAnalysis

class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight.view(1, -1, 1, 1)
        if self.use_bias:
            x = x + self.bias.view(1, -1, 1, 1)
        return x
    
def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight

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
    
class reweight_mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    
    
class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))
    
class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=[20], weight_resize=False,
                 **kwargs):
        super().__init__()
        # size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = MLP(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape
        
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x
    
class feq_WeightGenerator(nn.Module):
    # 输出应该为B*H*W*(级联解码器下一层通道数 × embedding)
    # in_dim表示当前输入特征维度 out_dim为下一层输入维度
    def __init__(self, in_dim: int = 256, out_dim: int = 256 ,embed_dim: int = 256,size:list = [120,160],groups:int = 8, num_filters:int = 6, bias:bool = True,reweight_expansion_ratio:int = 2, expansion_ratio:int =2, weight_resize:bool = False):
        super().__init__()
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.med_channels = int(expansion_ratio * in_dim)
        # self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(out_dim, self.med_channels, bias=bias)
        self.act1 = StarReLU()
        self.reweight = reweight_mlp(in_dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.weight_resize = weight_resize
        self.act2 = StarReLU()
        self.pwconv2 = nn.Linear(self.med_channels, embed_dim, bias=bias)
        
        ######################################################
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,1),
            nn.Conv2d(in_dim//2, in_dim//2,3,padding=1,groups=in_dim//2),
            nn.BatchNorm2d(in_dim//2),
            StarReLU()
        )
        self.channle_expand = nn.Conv2d(in_dim//2, out_dim * embed_dim, 1,groups = groups)
        
    def forward(self, x: Tensor,y: Tensor) -> Tensor:
        B, _, H, W = y.shape
        routeing = self.reweight(x.mean(dim=(2, 3))).view(B, self.num_filters,-1).softmax(dim=1)
        x = self.feature_fuse(x) # torch.Size([8, 256, 20, 15])
        feq = self.pwconv1(y.permute(0,2,3,1))
        feq = self.act1(feq)
        feq = feq.to(torch.float32)
        feq = torch.fft.rfft2(feq, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, feq.shape[1],
                                                    feq.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
            
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, feq.shape[1], feq.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        feq = feq * weight
        feq = torch.fft.irfft2(feq, s=(H, W), dim=(1, 2), norm='ortho')

        feq = self.act2(feq)
        feq = self.pwconv2(feq)
        
        W_Decoder = self.channle_expand(x) # B*(out_dim*embedding)*H*W
        b, c, h, w = y.shape
       
        fh, fw = W_Decoder.shape[-2:]
        
        ph, pw = y.shape[-2] // fh, y.shape[-1] // fw
        W_Decoder = W_Decoder.permute(0, 2, 3, 1).reshape(b * fh * fw * self.embed_dim, self.out_dim,1,1)
       
        y = y.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw) #
        y = F.conv2d(y, W_Decoder, groups=b * fh * fw)
        y = y.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)
       
        return (y * feq.permute(0,3,1,2)) + y   
     
class feq_WeightGenerator_V2(nn.Module):
    def __init__(self, in_dim: int = 256, out_dim: int = 256 ,embed_dim: int = 256,size:list = [120,160],groups:int = 8, num_filters:int = 6, bias:bool = True,reweight_expansion_ratio:float = 2, expansion_ratio:float =2, weight_resize:bool = False):
        super().__init__()
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.med_channels = int(expansion_ratio * in_dim)
        # self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(out_dim, self.med_channels, bias=bias)
        self.act1 = nn.GELU()
        self.reweight = reweight_mlp(int(in_dim//2), int((reweight_expansion_ratio * in_dim)//2), num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.weight_resize = weight_resize
        # self.act2 = StarReLU()
        self.pwconv2 = nn.Linear(self.med_channels, embed_dim, bias=bias)
        
        ######################################################
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,1),
            nn.Conv2d(in_dim//2, in_dim//2,3,padding=1,groups=in_dim//2),
            LayerNormGeneral(in_dim//2,(1,)),
            StarReLU()
        )
        self.channle_expand = nn.Conv2d(in_dim//2, out_dim * embed_dim, 1,groups = groups)
        
        # self.reweight = MLP(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        
    @staticmethod    
    def high_pass_filter(feq, D0 = 35):
        B, H_freq, W_freq, C = feq.shape
        
        # 创建 H_freq × W_freq 的网格坐标
        y, x = torch.meshgrid(torch.arange(H_freq), torch.arange(W_freq), indexing="ij")
        
        # 计算频谱距离 (以中心为基准)
        center = ((H_freq - 1) // 2, 0)  # rfft2 沿着 W 轴的频率只有一半
        distance = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
        
        # 构造高通滤波掩码，低频位置为 0，高频为 1
        high_pass_mask = (distance > D0).float().to(feq.device)
        
        # 进行高通滤波
        feq_filtered = feq * high_pass_mask[None, :, :, None]
        
        return feq_filtered
        
    def forward(self, x: Tensor,y: Tensor) -> Tensor:
        B, _, H, W = y.shape
        x = self.feature_fuse(x) # torch.Size([8, 256, 20, 15])
        routeing = self.reweight(x.mean(dim=(2, 3))).view(B, self.num_filters,-1)
        feq = self.pwconv1(y.permute(0,2,3,1))
        
        feq = self.act1(feq)
        feq = feq.to(torch.float32)
        feq = torch.fft.rfft2(feq, dim=(1, 2), norm='ortho')
        
        complex_weights = torch.view_as_complex(self.complex_weights)
            
        routeing = routeing.to(torch.complex64)
        
        weight = F.sigmoid(torch.einsum('bfc,hwf->bhwc', routeing, complex_weights))
        feq = self.high_pass_filter(feq)
        feq = feq * weight
        
        # print(feq.shape)
        feq = torch.fft.irfft2(feq, s=(H, W), dim=(1, 2), norm='ortho')
        # feq = self.act2(feq)
        feq = self.pwconv2(feq)
        W_Decoder = self.channle_expand(x) # B*(out_dim*embedding)*H*W
        
        b, c, h, w = y.shape
        
        fh, fw = W_Decoder.shape[-2:]
        
        ph, pw = y.shape[-2] // fh, y.shape[-1] // fw
        W_Decoder = W_Decoder.permute(0, 2, 3, 1).reshape(b * fh * fw * self.embed_dim, self.out_dim,1,1)
        
        y = y.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw) #
        y = F.conv2d(y, W_Decoder, groups=b * fh * fw)
        y = y.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)
        
        return y * feq.permute(0,3,1,2) + y
    
class WeightGenerator_V4(nn.Module):
    # 输出应该为B*H*W*(级联解码器下一层通道数 × embedding)
    # in_dim表示当前输入特征维度 out_dim为下一层输入维度
    def __init__(self, in_dim: int = 256, out_dim: int = 256 ,embed_dim: int = 256, groups:int=8,num_filters = 8):
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.channle_reduce = nn.Conv2d(in_dim,in_dim//4,kernel_size=3,padding=1,groups=in_dim//4)
        self.dwconv3 = nn.Conv2d(in_dim//4,in_dim//4,kernel_size=3,padding=1,groups=in_dim//4)
        self.dwconv7 = nn.Conv2d(in_dim//4,in_dim//4,kernel_size=5,padding=2,groups=in_dim//4)
        self.groups = groups
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,1),
            nn.BatchNorm2d(in_dim//2),
            StarReLU()
        )
        self.linear1 = nn.Linear(in_dim//(2*groups), out_dim * embed_dim//groups)
        # self.channle_expand = nn.Conv2d(in_dim//2, out_dim * embed_dim, 1, groups = groups)
        # self.act = StarReLU()
        
    def forward(self, x: Tensor,y: Tensor) -> Tensor:
        B, C, H, W = x.shape
        gate = self.gate(x)
        x = self.channle_reduce(x) # torch.Size([8, 256, 20, 15])
        x = torch.concat([self.dwconv3(x), self.dwconv7(x)], dim = 1)
        x = x * gate # shape: B,C//2, H, W
        x = x.reshape(B, self.groups, -1, H, W).permute(0, 3, 4, 1, 2)  # shape: B, H, W, G,C//(2*G), 
        W_Decoder = self.linear1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B, H, W, G, out_dim * embed_dim//groups,
        # W_Decoder = self.channle_expand(x) # B*(out_dim*embedding)*H*W
        b, c, h, w = y.shape
        fh, fw = W_Decoder.shape[-2:]
        ph, pw = y.shape[-2] // fh, y.shape[-1] // fw
        W_Decoder = W_Decoder.permute(0, 2, 3, 1).reshape(b * fh * fw * self.embed_dim, self.out_dim,1,1)
        y = y.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw) #
        y = F.conv2d(y, W_Decoder, groups=b * fh * fw)
        y = y.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)
        return y
        
class WeightGenerator_V3(nn.Module):
    # 输出应该为B*H*W*(级联解码器下一层通道数 × embedding)
    # in_dim表示当前输入特征维度 out_dim为下一层输入维度
    def __init__(self, in_dim: int = 256, out_dim: int = 256 ,embed_dim: int = 256, groups:int=8,num_filters = 8):
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.channle_reduce = nn.Conv2d(in_dim,in_dim//4,kernel_size=3,padding=1,groups=in_dim//4)
        self.dwconv3 = nn.Conv2d(in_dim//4,in_dim//4,kernel_size=3,padding=1,groups=in_dim//4)
        self.dwconv7 = nn.Conv2d(in_dim//4,in_dim//4,kernel_size=5,padding=2,groups=in_dim//4)
        self.groups = groups
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,1),
            nn.BatchNorm2d(in_dim//2),
            StarReLU()
        )
        self.channle_expand = nn.Conv2d(in_dim//2, out_dim * embed_dim, 1, groups = groups)
        
    def forward(self, x: Tensor,y: Tensor) -> Tensor:
        B, _, H, W = x.shape
        gate = self.gate(x)
        x = self.channle_reduce(x) # torch.Size([8, 256, 20, 15])
        x = torch.concat([self.dwconv3(x), self.dwconv7(x)], dim = 1)
        x = x * gate # shape: B,C//2, H, W
        W_Decoder = self.channle_expand(x) # B*(out_dim*embedding)*H*W
        b, c, h, w = y.shape
        fh, fw = W_Decoder.shape[-2:]
        ph, pw = y.shape[-2] // fh, y.shape[-1] // fw
        W_Decoder = W_Decoder.permute(0, 2, 3, 1).reshape(b * fh * fw * self.embed_dim, self.out_dim,1,1)
        y = y.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw) #
        y = F.conv2d(y, W_Decoder, groups=b * fh * fw)
        y = y.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)
        return y
    
class WeightGenerator_V2(nn.Module):
    # 输出应该为B*H*W*(级联解码器下一层通道数 × embedding)
    # in_dim表示当前输入特征维度 out_dim为下一层输入维度
    def __init__(self, in_dim: int = 256, out_dim: int = 256 ,embed_dim: int = 256, groups:int=8,num_filters = 8):
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,1),
            nn.Conv2d(in_dim//2, in_dim//2,3,padding=1,groups=in_dim//2), # modify
            nn.BatchNorm2d(in_dim//2),
            StarReLU()
        )
        self.channle_expand = nn.Conv2d(in_dim//2, num_filters, 1)
        self.W_base = nn.Parameter(
            torch.randn(out_dim*embed_dim, num_filters, dtype=torch.float32) * 0.02)
        self.act = StarReLU()
        
    def forward(self, x: Tensor,y: Tensor) -> Tensor:
        B, _, H, W = x.shape
        x = self.feature_fuse(x) # torch.Size([8, 256, 20, 15])
        # B ,num_filters,H*W
        W_Decoder = self.act(self.channle_expand(x)) # B*(out_dim*embedding)*H*W
        W_Decoder = torch.einsum('bfhw,ef->behw', W_Decoder, self.W_base)
        b, c, h, w = y.shape
        fh, fw = W_Decoder.shape[-2:]
        ph, pw = y.shape[-2] // fh, y.shape[-1] // fw
        W_Decoder = W_Decoder.permute(0, 2, 3, 1).reshape(b * fh * fw * self.embed_dim, self.out_dim,1,1)
        # print(y.shape)
        # RuntimeError: shape '[3, 128, 25, 4, 19, 3]' is invalid for input of size 2880000
        y = y.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw) #
        y = F.conv2d(y, W_Decoder, groups=b * fh * fw)
        y = y.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)
        # print(W_Decoder.shape)
        # torch.Size([1, 320, 512, 20, 15])
        return y
    
class WeightGenerator(nn.Module):
    def __init__(self, in_dim: int = 256, out_dim: int = 256 ,embed_dim: int = 256,groups:int=8):
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,1),
            nn.Conv2d(in_dim//2, in_dim//2,3,padding=1,groups=in_dim//2), # modify
            nn.BatchNorm2d(in_dim//2),
            StarReLU()
        )
        self.channle_expand = nn.Conv2d(in_dim//2, out_dim * embed_dim, 1, groups = groups)
        
    def forward(self, x: Tensor,y: Tensor) -> Tensor:
        B, _, H, W = x.shape
        x = self.feature_fuse(x) # torch.Size([8, 256, 20, 15])
        W_Decoder = self.channle_expand(x) # B*(out_dim*embedding)*H*W
        b, c, h, w = y.shape
        fh, fw = W_Decoder.shape[-2:]
        ph, pw = y.shape[-2] // fh, y.shape[-1] // fw
        W_Decoder = W_Decoder.permute(0, 2, 3, 1).reshape(b * fh * fw * self.embed_dim, self.out_dim,1,1)
        y = y.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw) #
        y = F.conv2d(y, W_Decoder, groups=b * fh * fw)
        y = y.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)
        return y

class DynamicsHead(nn.Module):
     
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.linear_c4 = MLP(dims[3], embed_dim)
        groups_list = [8,8,16]
        self.add_module(f"WeightGenerator_0", feq_WeightGenerator_V2(dims[3] ,dims[0], embed_dim= embed_dim, groups = 8, num_filters = 8, bias = True, reweight_expansion_ratio = 2, expansion_ratio=0.25, weight_resize = False))
        for i in range(1,len(dims)-1):# 0 1 2
            self.add_module(f"WeightGenerator_{i}", WeightGenerator(dims[3] ,dims[i], embed_dim,groups=groups_list[i]))
        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)
        print("embedding:"+str(embed_dim))

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shapeZ
        outs = [F.interpolate(self.linear_c4(features[3]).permute(0, 2, 1).reshape(B, -1, *features[3].shape[-2:]), size=(H, W), mode='bilinear', align_corners=False)]
        for i in range(2,-1,-1): #0 1 2 
            cf = eval(f"self.WeightGenerator_{i}")(features[3],features[i])  
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg

if __name__ == '__main__':
    num_minibatch = 1
    gpu=1
    ly_4 = torch.randn(num_minibatch, 512, 20, 15).cuda(gpu)
    ly_1 = torch.randn(num_minibatch, 64, 160, 120).cuda(gpu)
    # model.eval()
    model = feq_WeightGenerator(in_dim = 512, out_dim = 64, embed_dim= 512, groups = 8, num_filters = 6, bias = True, reweight_expansion_ratio = 2, expansion_ratio=0.25, weight_resize = False).cuda(gpu)
    model.eval()
    print(flop_count_table(FlopCountAnalysis(model, (ly_4,ly_1))))     
    print("%s | %s" % ("Params(M)", "FLOPs(G)"))