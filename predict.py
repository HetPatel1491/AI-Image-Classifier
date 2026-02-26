import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ImageClassifierNet
import os

def predict_image(image_path):
    # 1. Define labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. Load the trained brain
    net = ImageClassifierNet()
    net.load_state_dict(torch.load('image_classifier.pth'))
    net.eval() # Set to evaluation mode

    # 3. Prepare the image (must match training exactly)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # Add a 'batch' dimension

    # 4. Make the guess
    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)
        
    print(f'--- Result: The AI thinks this is a {classes[predicted[0]]} ---')

if __name__ == '__main__':
    # Place a test image in your folder and put the name here!
    test_img = "test.jpg" 
    
    if os.path.exists(test_img):
        predict_image(test_img)
    else:
        print(f"Please put an image named '{test_img}' in your folder first!")