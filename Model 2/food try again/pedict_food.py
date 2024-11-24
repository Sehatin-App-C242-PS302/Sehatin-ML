import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tkinter import Tk, filedialog

# Fungsi untuk membaca kelas makanan dari folder dataset
def load_class_indices(dataset_path):
    """Membaca nama kelas dari folder dalam dataset."""
    classes = sorted(os.listdir(dataset_path))  # Folder dataset berisi subfolder kelas
    class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
    return class_indices, classes

# Fungsi untuk prediksi berdasarkan model dan input gambar
def predict_food_interactive(model, class_indices):
    """Memprediksi kelas makanan berdasarkan input gambar."""
    Tk().withdraw()  # Sembunyikan jendela tkinter utama
    file_path = filedialog.askopenfilename(title="Pilih gambar makanan", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        print("Tidak ada file gambar yang dipilih!")
        return

    # Preprocessing gambar
    img = load_img(file_path, target_size=(224, 224))  # Sesuaikan ukuran input model
    x = img_to_array(img) / 255.0  # Normalisasi gambar
    x = np.expand_dims(x, axis=0)  # Tambahkan dimensi batch
    images = np.vstack([x])

    # Prediksi gambar
    predictions = model.predict(images, batch_size=1)
    predicted_class_index = np.argmax(predictions)

    # Cari nama kelas berdasarkan indeks
    class_names = list(class_indices.keys())
    label_name = class_names[predicted_class_index]

    # Tampilkan hasil
    print(f"Gambar: {file_path}")
    print(f"Prediksi: {label_name}")

# Path dataset dan model
dataset_path = "C:/Users/valen/Downloads/Sehatin-ML-main/Sehatin-ML/Model 2/food try again/food_datasett" # Ganti dengan path folder dataset Anda
model_path = "C:/Users/valen/Downloads/Sehatin-ML-main/Sehatin-ML\Model 2/food try again/best_model.h5"  # Ganti dengan path model yang sudah disimpan

# Memuat kelas makanan
class_indices, classes = load_class_indices(dataset_path)

# Memuat model yang telah disimpan
model = load_model(model_path)

# Memulai proses prediksi
predict_food_interactive(model, class_indices)