import cv2
import numpy as np
from tensorflow.keras.models import load_model

model= load_model("livercell_classifier3.keras")

def predict_image(image_path):
    
    img = cv2.imread(image_path)
    
    if img is None: 
        print("Image not found")
        return
    
    img = cv2.resize(img, (64, 64))
    
    img = img.astype(np.float32)/255.0
    
    img = np.expand_dims(img,axis=0)
    
    probability = model.predict(img, verbose=0)[0][0]
    
    print(f"Prediction score: {probability:.4f}")
    
    if probability >= 0.6:
        print("Result: Damaged Cell")
    else:
        print("result: Healthy Cell")
        
if __name__ == "__main__":
    image_path = input("enter image path: ")
    predict_image(image_path)
    
