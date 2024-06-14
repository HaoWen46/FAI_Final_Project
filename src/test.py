import torch

checkpoint = torch.load('./checkpoint.pt')
parameters = checkpoint['estimator']
torch.save(parameters, './parameters.pt')