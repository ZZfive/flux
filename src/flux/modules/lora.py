import torch
from torch import nn


def replace_linear_with_lora(  # 将module中的所有Linear层替换为LinearLora层
    module: nn.Module,
    max_rank: int,  # Lora最大的秩
    scale: float = 1.0,  # Lora的缩放因子
) -> None:
    for name, child in module.named_children():  # 遍历所有子模块
        if isinstance(child, nn.Linear):  # 如果当前模块为线性层Linear
            new_lora = LinearLora(
                in_features=child.in_features,  # 保持输入维度不变
                out_features=child.out_features,  # 保持输出维度不变
                bias=child.bias,  # 保持偏置不变
                rank=max_rank,  # 设置Lora的秩
                scale=scale,  # 设置Lora的缩放因子
                dtype=child.weight.dtype,  # 保持数据类型不变
                device=child.weight.device,  # 保持设备位置不变
            )  # 初始化一个LinearLora层
            # 将原始Linear层中的权重复制给新初始化的LinearLora层
            new_lora.weight = child.weight
            new_lora.bias = child.bias if child.bias is not None else None

            setattr(module, name, new_lora)  # 将新初始化的LinearLora层替换到module中
        else:
            replace_linear_with_lora(
                module=child,
                max_rank=max_rank,
                scale=scale,
            )  # 递归替换子模块中的Linear层


class LinearLora(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        lora_bias: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )

        assert isinstance(scale, float), "scale must be a float"

        self.scale = scale
        self.rank = rank
        self.lora_bias = lora_bias
        self.dtype = dtype
        self.device = device

        if rank > (new_rank := min(self.out_features, self.in_features)):  # 确保rank不超过输入、输出维度的最小值，就是要同时小于out_features和in_features
            self.rank = new_rank
        
        # 初始化Lora的A和B矩阵
        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=self.rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.lora_B = nn.Linear(
            in_features=self.rank,
            out_features=out_features,
            bias=self.lora_bias,
            dtype=dtype,
            device=device,
        )

    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        输入 x
          │
          ├─────────────────┐
          │                 │
        原始线性层          LoRA路径
          │                 │
          │            降维(lora_A)
          │                 │
          │             rank维度
          │                 │
          │            升维(lora_B)
          │                 │
          │              缩放(scale)
          │                 │
          └────────┬────────┘
                合并
                │
                输出
        '''
        base_out = super().forward(input)  # 先进行原始的线性变换

        _lora_out_B = self.lora_B(self.lora_A(input))  # 计算Lora的输出
        lora_update = _lora_out_B * self.scale  # 计算Lora的更新量

        return base_out + lora_update
