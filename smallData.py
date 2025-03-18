import torchvision
dataset = torchvision.datasets.Kinetics(root="./data/kinetics", frames_per_clip=16, download=True)
