import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from fvcore.nn import flop_count_table, FlopCountAnalysis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass

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

# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda
    
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True
            # print("A:shape:"+ str(A.shape))
            # print(u.shape)
            # torch.Size([96, 8]) torch.Size([1, 1843200, 384])
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossChannelScan_multimodal(torch.autograd.Function):
        
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W  B, S, D, C
            B, S, D, C = x_rgb.shape
            ctx.shape = (B, S, D, C)
            xs_fuse = x_rgb.new_empty((B*S, 2, D, 2 * C))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(0, 1), x_e.flatten(0, 1)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            "理论上来说在前向传播中使用stack函数而不是new一个新张量可以避免反向传播的手动实现"
            # out: (b, 2, d, l)
            B, S, D, C = ctx.shape
            L = 2 * C
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B*S, D, 2 * C
            # get B, d, H*W
            return ys[:, :, 0:C].view(B, -1, D, C), ys[:, :, C:2*C].view(B, -1, D, C)
        
    class CrossChannleMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape 
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L//2], ys[:, :, L//2:L]
        
        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, D, C = x1.shape
            xs = x1.new_empty((B, 2, D, 2*C))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, D, 2*C)
            return xs, None, None

    def selective_scan_1d(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)
        
        # y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def cross_selective_scan_multimodal_k1(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L//2] + ys[:, 0, :, L//2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)
        
        return y

    def cross_selective_scan_multimodal_channle(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        "B*N"
        # (B*S)*Head*C
        B, S, D, C = x_rgb.shape
        # A = (K * D, N)
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * C

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossChannelScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B*S, 2, D, 2 * C
        # print(x_fuse.shape)
        # print(x_proj_weight.shape)
        # torch.Size([19200, 2, 48, 384]) torch.Size([2, 11, 48])
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B*S, -1, L).to(torch.float)
        dts = dts.contiguous().view(B*S, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B*S, K, -1, 2*C)
        
        y_rgb, y_e = CrossChannleMerge_multimodal.apply(ys)
        # print(y_rgb.shape)
        # todo check
        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, S, D, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, S, D, -1)

        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)
        
        return y_rgb, y_e


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try: 
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[1].debugName().startswith("dts") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    assert inputs[4].debugName().startswith("Cs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False
  
# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y
    
class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
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
    
class ChannelMambaFusionBlock_Mask(nn.Module):
    '''
    ChannelMambaFusion
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.1,
        d_state: int = 12, 
        dt_rank: Any = "auto",
        ssm_ratio = 2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.op = ChannelMB_SS2D_Mask(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state = d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.in_norm1 = norm_layer(hidden_dim)
            self.in_norm2 = norm_layer(hidden_dim)
            self.norm1 = norm_layer(hidden_dim)
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp_rgb = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
            self.mlp_e = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        # 输入为: B, C, H, W
        y_rgb, y_e = self.op(self.in_norm1(x_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.in_norm2(x_e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        if self.mlp_branch:
            y_rgb = y_rgb + self.drop_path(self.mlp_rgb(self.norm1(y_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) # FFN
            y_e = y_e + self.drop_path(self.mlp_e(self.norm2(y_e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) # FFN
        return y_rgb, y_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)
        
class ChannelMambaFusionBlock(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.1,
        d_state: int = 12, # 已经修改
        dt_rank: Any = "auto",
        ssm_ratio = 2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = ChannelMB_SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state = d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.in_norm1 = norm_layer(hidden_dim)
            self.in_norm2 = norm_layer(hidden_dim)
            self.norm1 = norm_layer(hidden_dim)
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp_rgb = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
            self.mlp_e = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor, idx:torch.Tensor):
        # 输入为: B, C, H, W
        y_rgb, y_e = self.op(self.in_norm1(x_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.in_norm2(x_e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), idx)
        if self.mlp_branch:
            y_rgb = y_rgb + self.drop_path(self.mlp_rgb(self.norm1(y_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) # FFN
            y_e = y_e + self.drop_path(self.mlp_e(self.norm2(y_e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) # FFN
        return y_rgb, y_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor,idx:torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e, idx)
        else:
            return self._forward(x_rgb, x_e, idx)
        
class ChannelMB_SS2D_Mask(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model = 96,
        d_state = 16,
        ssm_ratio = 2,
        dt_rank= "auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias = True,
        # ======================
        dropout=0.1,
        bias = False,
        conv_embedding = True,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        head: int = 16,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(head / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.expand_d = int(self.expand * self.d_model)
        self.d_inner = int(self.expand * self.d_model) // head
        self.dt_rank = math.ceil(head / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Conv2d(self.d_model, self.expand_d,1, bias=bias, groups = self.d_model // 4,**factory_kwargs)
        self.in_proj_modalx = nn.Conv2d(self.d_model, self.expand_d,1,bias=bias, groups = self.d_model // 4 ,**factory_kwargs)
        self.head = head
        # self.head_rgbproj = nn.Linear(self.d_inner, self.d_inner*head)
        # self.head_modalxproj = nn.Linear(self.d_inner, self.d_inner*head)
        self.conv_embedding = conv_embedding
        # conv =======================================
        if self.d_conv > 1:
            if conv_embedding:
                # 应该改为1*1卷积和dw conv的组合
                self.conv2d = nn.Conv2d(
                    in_channels = self.expand_d,
                    out_channels = self.expand_d,
                    groups=self.expand_d,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    # stride = 2,
                    **factory_kwargs,
                )
                self.conv2d_modalx = nn.Conv2d(
                    in_channels = self.expand_d,
                    out_channels = self.expand_d,
                    groups=self.expand_d,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    # stride = 2,
                    **factory_kwargs,
                )
            else:
                self.conv2d = nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    groups=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    
                    **factory_kwargs,
                )
                self.conv2d_modalx = nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    groups=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.head, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.head, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.head, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.head, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj_rgb = nn.Conv2d(self.expand_d ,self.d_model,1,bias=bias,groups=self.d_model // 4, **factory_kwargs)
        self.out_proj_modalx = nn.Conv2d(self.expand_d, self.d_model,1, bias=bias,groups=self.d_model // 4, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization (K * D, N) 
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
     
    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_channle(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version, 
            nrows=nrows,
        )

    def forward(self, input_rgb: torch.Tensor, input_e: torch.Tensor):
        "x_rgb and x_e:B*C*H*W"
        "对channle数量进行扩张维度为 x_rgb and x_e:B*d_inner*H*W"
        x_rgb = self.in_proj(input_rgb)
        x_e = self.in_proj_modalx(input_e)
        if self.d_conv > 1:
            if self.conv_embedding:
                # 之前输入应该为 B*H*W*C ->对应B*L*C
                "x_rgb and x_e:B*d_inner*H*W"
                B, C, H, W = x_rgb.shape
               
                x_rgb_conv = self.act(self.conv2d(x_rgb)) # (b, d, h, w)
                x_e_conv = self.act(self.conv2d_modalx(x_e)) # (b, d, h, w)
                # print(x_rgb_conv.shape)
                x_rgb_head =  x_rgb_conv.permute(0, 2, 3, 1).contiguous().view(B, H*W, self.head, -1) #(b, h, w,d*group)
                x_e_head =  x_e_conv.permute(0, 2, 3, 1).contiguous().view(B, H*W, self.head, -1)
                
                y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_head, x_e_head) # b, d, h, w -> b, h, w, d
                
                y_rgb = self.out_proj_rgb(y_rgb.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
                y_e = self.out_proj_modalx(y_e.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
                y_rgb = self.dropout_rgb(y_rgb) + input_rgb
                y_e = self.dropout_e(y_e) + input_e
                
        return y_rgb, y_e 
    
class ChannelMB_SS2D(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model = 96,
        d_state = 16,
        ssm_ratio = 2,
        dt_rank= "auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias = True,
        # ======================
        dropout=0.1,
        bias = False,
        conv_embedding = True,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        head: int = 16,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(head / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.expand_d = int(self.expand * self.d_model)
        self.d_inner = int(self.expand * self.d_model) // head
        self.dt_rank = math.ceil(head / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Conv2d(self.d_model, self.expand_d,1, bias=bias, groups = self.d_model // 4,**factory_kwargs)
        self.in_proj_modalx = nn.Conv2d(self.d_model, self.expand_d,1,bias=bias, groups = self.d_model // 4 ,**factory_kwargs)
        self.head = head
        # self.head_rgbproj = nn.Linear(self.d_inner, self.d_inner*head)
        # self.head_modalxproj = nn.Linear(self.d_inner, self.d_inner*head)
        self.conv_embedding = conv_embedding
        # conv =======================================
        if self.d_conv > 1:
            if conv_embedding:
                # 应该改为1*1卷积和dw conv的组合
                self.conv2d = nn.Conv2d(
                    in_channels = self.expand_d,
                    out_channels = self.expand_d,
                    groups=self.expand_d,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    # stride = 2,
                    **factory_kwargs,
                )
                self.conv2d_modalx = nn.Conv2d(
                    in_channels = self.expand_d,
                    out_channels = self.expand_d,
                    groups=self.expand_d,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    # stride = 2,
                    **factory_kwargs,
                )
            else:
                self.conv2d = nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    groups=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    
                    **factory_kwargs,
                )
                self.conv2d_modalx = nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    groups=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.head, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.head, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.head, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.head, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj_rgb = nn.Conv2d(self.expand_d ,self.d_model,1,bias=bias,groups=self.d_model // 4, **factory_kwargs)
        self.out_proj_modalx = nn.Conv2d(self.expand_d, self.d_model,1, bias=bias,groups=self.d_model // 4, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization (K * D, N) 
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
     
    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_channle(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version, 
            nrows=nrows,
        )

    def forward(self, input_rgb: torch.Tensor, input_e: torch.Tensor, idx: torch.Tensor):
        """
        input_rgb, input_e: [B, C, H, W]
        idx: [B, N] 每个 batch 的稀疏采样索引，值在 [0, H*W)
        """
        B, _, H, W = input_rgb.shape
        N = H * W
        d_inner = self.in_proj.out_channels  # e.g., 64
        device = input_rgb.device

        # 投影到 shared embedding space
        x_rgb = self.in_proj(input_rgb)          # [B, d_inner, H, W]
        x_e = self.in_proj_modalx(input_e)       # [B, d_inner, H, W]

        if self.d_conv > 1 and self.conv_embedding:
            x_rgb = self.act(self.conv2d(x_rgb))         # [B, d_inner, H, W]
            x_e = self.act(self.conv2d_modalx(x_e))      # [B, d_inner, H, W]

            # reshape to [B, H*W, head, dim_head]
            x_rgb_flat = x_rgb.permute(0, 2, 3, 1).reshape(B, N, self.head, -1)
            x_e_flat = x_e.permute(0, 2, 3, 1).reshape(B, N, self.head, -1)

            # expand idx for gather
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.head, x_rgb_flat.shape[-1])  # [B, num_q, head, dim]

            # Gather sparse feature for attention
            x_rgb_sparse = torch.gather(x_rgb_flat, dim=1, index=idx_exp)  # [B, num_q, head, dim]
            x_e_sparse = torch.gather(x_e_flat, dim=1, index=idx_exp)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_sparse, x_e_sparse)  # [B, num_q, head, dim]

            # Scatter 回 full tensor
            y_rgb_full = torch.zeros_like(x_rgb_flat)  # [B, H*W, head, dim]
            y_e_full = torch.zeros_like(x_e_flat)
            y_rgb_full.scatter_(1, idx_exp, y_rgb)
            y_e_full.scatter_(1, idx_exp, y_e)

            y_rgb = y_rgb_full.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, d_inner, H, W]
            y_e = y_e_full.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb)) + input_rgb  # [B, C, H, W]
            out_e = self.dropout_e(self.out_proj_modalx(y_e)) + input_e
            return out_rgb, out_e
           
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

if __name__ == '__main__':
    modals = ['img', 'thermal'] 
    num_minibatch = 1
    gpu=1
    model = ChannelMambaFusionBlock(hidden_dim =  64, head = 32, ssm_ratio = 4,mlp_ratio = 2).cuda(gpu)
    x_rgb = torch.zeros(1, 64, 160, 120).cuda(gpu) 
    x_e = torch.zeros(1, 64, 160, 120).cuda(gpu)
    idx = generate_topk_cosine_mask(x_rgb, x_e, 0.1).cuda(gpu)
    # model(x_rgb, x_e, idx)
    from thop import profile
    total_ops, total_params  = profile(model, inputs=(x_rgb,x_e,idx), verbose=False)
    # print(flop_count_table(FlopCountAnalysis(model, (x_rgb, x_e, idx))))     
    print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    print("%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))    
