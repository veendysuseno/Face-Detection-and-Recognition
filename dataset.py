import numpy as np
import cv2

# Mengaktifkan kamera
cam = cv2.VideoCapture(0)

# Memuat classifier deteksi wajah
faceDetect = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Meminta user ID untuk dataset
id = input('Enter user ID: ')
sampleNum = 0  # Variabel untuk menghitung jumlah sampel yang diambil

while True:
    ret, img = cam.read()  # Membaca frame dari kamera
    if not ret:
        print("Error: Unable to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Mengubah frame menjadi grayscale
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # Mendeteksi wajah

    for (x, y, w, h) in faces:
        sampleNum += 1  # Increment jumlah sampel
        # Menyimpan wajah yang terdeteksi sebagai gambar di folder dataset
        cv2.imwrite(f"dataSet/User.{id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        
        # Menampilkan kotak persegi panjang di sekitar wajah
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Menampilkan gambar dengan wajah yang dideteksi
    cv2.imshow("Face", img)

    # Jika sudah menyimpan sama dengan 50 gambar, keluar dari loop
    if sampleNum == 50:
        break

    # Memberikan jeda untuk menampilkan frame
    if cv2.waitKey(1) == ord('q'):  # Pengguna bisa keluar dengan menekan tombol 'q'
        break

# Membersihkan sumber daya kamera dan menutup semua jendela OpenCV
cam.release()
cv2.destroyAllWindows()
