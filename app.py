import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image
from model import ImageClassifierNet
import time

# Get the absolute path of the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# CIFAR-10 Labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 1. Load the Model (The Brain)
net = ImageClassifierNet()
try:
    model_path = os.path.join(BASE_DIR, 'image_classifier.pth')
    # map_location ensures it works even if you don't have a GPU
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    print("\n" + "="*30)
    print("--- ✅ Model Loaded Successfully ---")
    print("="*30 + "\n")
except Exception as e:
    print(f"--- ❌ Error Loading Model: {e} ---")

# 2. Image Preprocessing Logic
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# 3. Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'result': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'No selected file'})

    # Save to a generic temp name to avoid path issues
    temp_path = os.path.join(BASE_DIR, "latest_upload.jpg")

    try:
        file.save(temp_path)
        # Short pause to prevent Windows/OneDrive file locking errors
        time.sleep(0.5) 

        tensor = transform_image(temp_path)
        
        with torch.no_grad():
            outputs = net(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            prob, index = torch.max(probabilities, 0)
            
            label = classes[index.item()].upper()
            confidence = round(prob.item() * 100, 1)

            # --- CONFIDENCE THRESHOLD (The "Honesty" Logic) ---
            if confidence < 50.0:
                final_result = f"🤔 I'm not 100% sure (only {confidence}% confidence), but it looks a bit like a {label}."
            else:
                final_result = f"✅ I am {confidence}% sure this is a {label}"
            
            print(f"--- 🎯 Prediction Made: {label} ({confidence}%) ---")

        return jsonify({'result': final_result})

    except Exception as e:
        print(f"--- ❌ Prediction Error: {e} ---")
        return jsonify({'result': f"Error: {str(e)}"})

# 4. Start the Server
if __name__ == '__main__':
    print("🚀 AI Server is launching...")
    print("🔗 Access your app at: http://127.0.0.1:8080")
    # debug=True allows the server to restart automatically if you change code
    app.run(host='127.0.0.1', port=8080, debug=True)