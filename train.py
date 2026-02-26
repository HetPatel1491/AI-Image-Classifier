import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ImageClassifierNet

def train():
    # 1. Data Prep with "Augmentation" (Helps AI learn better)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # 2. Initialize Model
    net = ImageClassifierNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 30
    print(f"Starting Training for {epochs} epochs...")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1} | Loss: {running_loss / len(trainloader):.3f}')

    # 4. Save the new brain
    torch.save(net.state_dict(), 'image_classifier.pth')
    print("--- Training Complete! Saved as image_classifier.pth ---")

if __name__ == '__main__':
    train()