from torch.profiler import profile, record_function, ProfilerActivity
import torch
# import torchvision.models as models
from train import mixing_noise

from model import Generator

# model = models.resnet18().cuda()
inputs = torch.randn(4, 3, 512, 512).cuda()
model = Generator(512, 512, 8, 2).cuda()

noise = mixing_noise(4,512,0.9,"cuda")

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(noise)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))