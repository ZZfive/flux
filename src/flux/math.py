import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """
    注意力机制
    q: query张量 [batch, heads, seq_len, head_dim]
    k: key张量 [batch, heads, seq_len, head_dim]
    v: value张量 [batch, heads, seq_len, head_dim]
    pe: 位置编码张量 [batch, 1, dim, 2, 2]
    """
    q, k = apply_rope(q, k, pe)  # 将预计算的rope旋转矩阵应用于q，k

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # 计算注意力
    x = rearrange(x, "B H L D -> B L (H D)")  # 将多头注意力组合回整体

    return x  # [batch, seq_len, heads*head_dim]


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)  # 将最后一维分成两部分
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]  # 旋转
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
