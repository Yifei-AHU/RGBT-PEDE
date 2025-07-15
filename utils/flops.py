import torch
from thop import profile
import clip

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 创建图像和文本的 dummy 输入
image = torch.randn(1, 3, 224, 224).cuda()  # 单张 224x224 RGB 图像
text = clip.tokenize(["a photo of a cat"]).cuda()  # 单个文本输入

# 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(image, text))

# 转换 FLOPs 为 G（Giga FLOPs）
flops_g = flops / 1e9
params_m = params / 1e6

print(f"FLOPs: {flops_g:.2f} G")
print(f"Parameters: {params_m:.2f} M")
