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
    # 计算每个位置的频率缩放因子；先生成序列 [0, 2, 4, ..., dim-2]，然后除以 dim，得到得到 [0, 2/dim, 4/dim, ..., (dim-2)/dim]
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)  # 计算最终的角频率  ω_i = 1/θ^(2i/dim)
    out = torch.einsum("...n,d->...nd", pos, omega)  # Einstein 求和约定计算位置和频率的外积，shape: [batch, seq_len, dim//2]
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)  # 构建旋转矩阵，shape: [batch, seq_len, dim//2, 4]
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # 重排列成矩阵形式，将最后一个维度4拆分成2*2，shape: [batch, seq_len, dim//2, 2, 2]
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # 输入的q、k张量的最后一维拆分为两个维度，相当于构建复述形式；[batch, heads, seq_len, head_dim] --> [batch, heads, seq_len, head_dim//2, 1, 2]，新增加的维度1是为了广播计算添加
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # 进行旋转变换；freqs_cis[..., 0]、freqs_cis[..., 1]是一个行数为2的列向量，xq_[..., 0]、xq_[..., 1]、xk_[..., 0]、xk_[..., 1]是一个列数为2的行向量
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)  # 将结果重排列回原来的形状
