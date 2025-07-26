from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):  # 可分别初始化clip和t5的文本编码器类
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")  # 判断是clip还是t5
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)  # 初始化clip的tokenizer
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)  # 初始化clip的文本编码器
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)  # 初始化t5的tokenizer
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)  # 初始化t5的文本编码器

        self.hf_module = self.hf_module.eval().requires_grad_(False)  # 设置为eval模式，并禁用梯度计算

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,  # 输入文本列表
            truncation=True,  # 允许截断
            max_length=self.max_length,  # 最大长度
            return_length=False,  # 不返回长度
            return_overflowing_tokens=False,  # 不返回溢出token
            padding="max_length",  # 填充到最大长度
            return_tensors="pt",  # 返回pytorch张量
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,  # 不使用注意力掩码
            output_hidden_states=False,  # 不输出隐藏状态
        )
        return outputs[self.output_key].bfloat16()  # 返回输出
