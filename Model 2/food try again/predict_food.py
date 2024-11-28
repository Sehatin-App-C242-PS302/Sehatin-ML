import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tkinter import Tk, filedialog

# Function to read food classes from dataset folder
def load_class_indices(dataset_path):
    """Membaca nama kelas dari folder dalam dataset."""
    classes = sorted(os.listdir(dataset_path)) 
    class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
    return class_indices, classes

# Function for prediction based on model and input image
def predict_food_interactive(model, class_indices):
    """Memprediksi kelas makanan berdasarkan input gambar."""
    Tk().withdraw() 
    file_path = filedialog.askopenfilename(title="Pilih gambar makanan", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        print("Tidak ada file gambar yang dipilih!")
        return

    # Preprocessing images
    img = load_img(file_path, target_size=(224, 224))  
    x = img_to_array(img) / 255.0 
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])

    predictions = model.predict(images, batch_size=1)
    predicted_class_index = np.argmax(predictions)

    class_names = list(class_indices.keys())
    label_name = class_names[predicted_class_index]

    print(f"Gambar: {file_path}")
    print(f"Prediksi: {label_name}")

# Path dataset and model
dataset_path = "./Model 2/food try again/food_datasett" 
model_path = "./Model 2/food try again/best_model.h5"

class_indices, classes = load_class_indices(dataset_path)

model = load_model(model_path)

predict_food_interactive(model, class_indices)