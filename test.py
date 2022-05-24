import torch
print(torch.cuda.is_available())
print(torch.__version__)
torch.zeros(1).cuda()