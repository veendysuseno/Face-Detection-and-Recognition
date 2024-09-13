import os
import cv2
import numpy as np
from PIL import Image

# Membuat recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path folder dataset
path = 'dataSet'

# Fungsi untuk mendapatkan gambar dan ID
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')  # Membuka gambar dan mengubahnya ke grayscale
        faceNp = np.array(faceImg, 'uint8')  # Mengubah gambar menjadi array numpy
        ID = int(os.path.split(imagePath)[-1].split('.')[1])  # Mendapatkan ID dari nama file
        faces.append(faceNp)  # Menambahkan gambar wajah ke list
        IDs.append(ID)  # Menambahkan ID ke list
        cv2.imshow('Training', faceNp)  # Menampilkan gambar saat dilatih (opsional)
        cv2.waitKey(10)  # Menunggu sebentar untuk setiap gambar
    return np.array(IDs), faces

# Mendapatkan ID dan gambar wajah dari dataset
Ids, faces = getImagesWithID(path)

# Melatih recognizer menggunakan gambar wajah dan ID
recognizer.train(faces, Ids)

# Membuat folder 'recognizer' jika belum ada
recognizer_path = 'recognizer'  # Ganti dengan path absolut jika perlu
if not os.path.exists(recognizer_path):
    os.makedirs(recognizer_path)

# Menyimpan hasil training ke file yml di folder 'recognizer'
output_file = os.path.join(recognizer_path, 'training_data.yml')
recognizer.save(output_file)

# Membersihkan semua window OpenCV
cv2.destroyAllWindows()
