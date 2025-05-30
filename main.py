import cv2
import numpy as np
import os
from src.model import SimpleCNN

def load_images(folder, label, size=(32, 32)):
    data = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        img = img / 255.0  # normalize
        data.append((img, label))
    return data

if __name__ == "__main__":
    model = SimpleCNN()

    # Load dataset
    data = load_images("data/senang", 0) + load_images("data/sedih", 1) + load_images("data/marah", 2)
    np.random.shuffle(data)

    # Train
    for epoch in range(10):
        total_loss = 0
        for img, label in data:
            loss = model.backward(img, label)
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
