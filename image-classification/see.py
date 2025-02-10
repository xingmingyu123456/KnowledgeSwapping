import torch  # 命令行是逐行立即执行的

content = torch.load('/home/xmy/code/classify-bird/forget_check/forget_1.pth')
content1 = torch.load('/home/xmy/code/classify-bird/forget_check/forget_final.pth')
# content2 = torch.load('/home/xmy/code/classify-bird/fisrt.pth')
# 不是完全相等有误差
for key in content:
    if not torch.allclose(content[key], content1[key], rtol=0, atol=0.0001):
        print(f"Mismatch found in key: {key}")
else:
    print("Both saved weights are identical.")
print(content["transformer.encoder.layer.0.ffn.fc1.weight"])
print(content1["transformer.encoder.layer.0.ffn.fc1.weight"])
# print(content)  # keys()
# # 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['model'])