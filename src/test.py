import torch

checkpoint = torch.load('./src/checkpoint.pt')
parameters = checkpoint['estimator']
torch.save(parameters, './src/parameters.pt')