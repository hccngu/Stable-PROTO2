import torch.nn.functional as F
import torch

output1 = torch.randint(0, 1, (900, 128))

output2 = torch.randint(0, 10, (900, 128))

euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

print(euclidean_distance.shape)
print(euclidean_distance)