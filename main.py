import os
import numpy as np
import mediapipe as mp
import cv2

# Model yolunu belirleme
model_path = 'C:/Users/omere/Downloads/gesture_recognizer.task'

# Dosyanın var olup olmadığını kontrol etme
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
else:
    print(f"Model dosyası bulundu: {model_path}")

# Gerekli sınıfları tanımlama
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Sonuçları yazdırma işlevi
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

# GestureRecognizer Seçeneklerini Tanımlama
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# GestureRecognizer oluşturma ve kullanma
with GestureRecognizer.create_from_options(options) as recognizer:
    # Web kamerasını başlat
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Web kamerası açılamadı")

    while cap.isOpened():
        ret, frame = cap.read()  # Bir kare oku
        if not ret:
            break

        # OpenCV görüntüsünü RGB'ye dönüştürme
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe Image sınıfını kullanarak RGB çerçeveyi oluşturma
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # GestureRecognizer ile el tespiti yapma
        recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Web kamerasında görüntüyü gösterme
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
