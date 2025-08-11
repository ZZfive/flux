import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope


class EmbedND(nn.Module):
    """N维位置编码模块
    
    参数:
        dim (int): 位置编码维度, 通常为64或128
        theta (int): RoPE旋转角度参数, 通常为10000
        axes_dim (list[int]): 每个轴的编码维度, 如[32,32]表示2D位置编码,每个维度32
    """
    def __init__(self, dim: int = 64, theta: int = 10000, axes_dim: list[int] = [32, 32]):
        super().__init__()
        self.dim = dim  # 位置编码总维度，等于axes_dim之和
        self.theta = theta  # RoPE旋转参数
        self.axes_dim = axes_dim  # 每个轴的编码维度，如[16,56,56]表示3D位置编码；所有维度之和应该和图像特征向量的维度相同
        
    def forward(self, ids: Tensor) -> Tensor:
        """前向传播
        
        参数:
            ids: shape为[batch_size, seq_len, n_axes]的位置索引张量，此处的seq_len是所有轴的编码维度之和
            
        返回:
            shape为[batch_size, 1, dim, 2, 2]的位置编码张量
        """
        n_axes = ids.shape[-1]  # 获取编码维度数，如[16,56,56]表示3D位置编码，则n_axes=3
        # 对每个轴单独应用RoPE编码，然后拼接到一起
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )  # 每个轴的编码在-3维度上拼接，因为最后两个维度是旋转矩阵
        # 增加head维度
        return emb.unsqueeze(1)  # [B, 1, D, 2, 2]，D为总编码维度，2x2为RoPE的旋转矩阵


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t  # 时间缩放因子可以放大时间，让编码更精细
    half = dim // 2  # sin、cos各占一半
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # 如果维度不是2的倍数，补充一个零向量
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 维度是3是因为输出为qkv拼接
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)  # 因为是自注意力，用x直接线性转换得到qkv拼接张量
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)  # 拆分得到qkv
        q, k = self.norm(q, k, v)  # 归一化
        x = attention(q, k, v, pe=pe)  # 注意力计算
        x = self.proj(x)  # 投影
        return x


@dataclass
class ModulationOut:
    shift: Tensor  # 偏移量，加性调制项
    scale: Tensor  # 缩放因子，乘性调制项
    gate: Tensor  # 门控系数，控制调制项的激活


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        """
        参数:
            img: 图像张量 [batch, seq_len, hidden_size], 图片latent
            txt: 文本张量 [batch, seq_len, hidden_size]，prompt经过T5 encoder后的嵌入
            vec: 调制向量 [batch, hidden_size], 是prompt经过clip、时间经过位置编码，guidance经过位置编码后的拼接张量
            pe: 位置编码张量 [batch, 1, dim, 2, 2]
        
        返回:
            img: 处理后的图像张量 [batch, seq_len, hidden_size]
            txt: 处理后的文本张量 [batch, seq_len, hidden_size]
        """
        # 分别预测图片和文本的调制项
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)  # 归一化
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift  # 调制
        img_qkv = self.img_attn.qkv(img_modulated)  # 单独使用图片自注意力模块中的qkv子模块从拼接在一起的qkv
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)  # 拆分
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)  # 单独使用图片自注意力模块中的归一化模块

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)  # 文本和图片的q拼接
        k = torch.cat((txt_k, img_k), dim=2)  # 文本和图片的k拼接
        v = torch.cat((txt_v, img_v), dim=2)  # 文本和图片的v拼接

        attn = attention(q, k, v, pe=pe)  # 注意力计算  
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]  # 拆分出文本和图片的注意力结果

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)  # 先单独使用图片自注意力模块中的投影层转换，再乘上图片调制项的门控系数，最后加上图片调制项的偏移量
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)  # 与图片调制项中的第二组调制参数组合

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)  # 并行线性层，qkv转换时同时将mlp输入也预测处理
        # proj and mlp_out 
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)  # 并行线性层，最后转换注意力计算时，直接和mlp的输出拼接一并处理

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        """
        参数:
            x: 输入张量 [batch, seq_len, hidden_size]，经过双流注意力模块后输出的image latent和prompt latent拼接后的张量
            vec: 调制向量 [batch, hidden_size], 是prompt经过clip、时间经过位置编码，guidance经过位置编码后的拼接张量
            pe: 位置编码张量 [batch, 1, dim, 2, 2]
        
        返回:
            x: 处理后的张量 [batch, seq_len, hidden_size]
        """
        mod, _ = self.modulation(vec)  # 调制，只预测一组调制参数
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift  # 调制
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)  # 拆分

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
