import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("--- Step 1: Script started successfully! ---")

# 1. Prepare the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("--- Step 2: Downloading data (This might take a minute)... ---")

# 2. Download the CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("--- Step 3: Data loaded! Grabbing 4 random images... ---")

# 3. Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(f"Labels found: {[classes[labels[j]] for j in range(4)]}")

# 4. Function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print("--- Step 4: Displaying window... CLOSE THE WINDOW TO CONTINUE ---")
    plt.show()

imshow(torchvision.utils.make_grid(images))
print("--- Script Finished! ---")