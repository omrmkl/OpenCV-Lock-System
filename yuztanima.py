import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import mediapipe as mp
import os
import threading
import time

root = tk.Tk()
root.title("Yüz ve El Hareketi Tanıma Sistemi")
root.geometry("1000x700")
root.configure(bg="#2c3e50")

header = Label(root, text="Yüz ve El Hareketi Tanıma Sistemi", font=("Helvetica", 24, "bold"), bg="#1abc9c", fg="white",
               pady=20)
header.pack(fill=tk.X)

video_frame = Frame(root, bg="#2c3e50")
video_frame.pack(side=tk.LEFT, padx=20, pady=20)

info_frame = Frame(root, bg="#34495e")
info_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)

camera_label = Label(video_frame, bg="#000")
camera_label.pack()

person_label = Label(info_frame, text="Kişi: Tanınmıyor", font=("Helvetica", 18), bg="#34495e", fg="white")
person_label.pack(pady=20)

gesture_label = Label(info_frame, text="El Hareketi: Tanınmıyor", font=("Helvetica", 18), bg="#34495e", fg="white")
gesture_label.pack(pady=20)

password_status_label = Label(info_frame, text="Şifre: -", font=("Helvetica", 18), bg="#34495e", fg="white")
password_status_label.pack(pady=20)

lock_status_label = Label(info_frame, text="Kilit Durumu: Kapalı", font=("Helvetica", 18), bg="#34495e", fg="red")
lock_status_label.pack(pady=20)

cap = cv2.VideoCapture(0)


def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame)


update_frame()

correct_sequence = ['Thumb_Up']
gesture_sequence = []
gesture_timeout = 2.0  # 2 seconds timeout
last_gesture_time = time.time()


def recognize():
    global last_gesture_time, gesture_sequence
    haarcascade_path = "C:/Users/omere/Downloads/haarcascade_frontalface_default(1).xml"
    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    if face_cascade.empty():
        raise IOError('Haarcascade xml file not found!')

    database_path = "C:/Users/omere/Downloads/yuzler/"

    model_path = 'C:/Users/omere/Downloads/gesture_recognizer.task'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        global last_gesture_time, gesture_sequence
        current_time = time.time()

        if result.gestures:
            gesture = result.gestures[0][0].category_name
            gesture_label.config(text=f"El Hareketi: {gesture}")

            if gesture != "None":
                # Update gesture sequence
                gesture_sequence.append(gesture)
                # Trim the sequence to the length of the correct sequence
                if len(gesture_sequence) > len(correct_sequence):
                    gesture_sequence.pop(0)

                # Check if the sequence matches the correct sequence
                if gesture_sequence == correct_sequence:
                    password_status_label.config(text="Şifre: *")
                    lock_status_label.config(text="Kilit Durumu: Açık", fg="green")
                else:
                    password_status_label.config(text="Şifre: -")
                    lock_status_label.config(text="Kilit Durumu: Kapalı", fg="red")

        else:
            gesture_label.config(text="El Hareketi: Tanınmıyor")
            lock_status_label.config(text="Kilit Durumu: Kapalı", fg="red")

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            face_detected = False
            for (x, y, w, h) in faces:
                face_detected = True
                face_img = frame[y:y + h, x:x + w].copy()

                try:
                    result = DeepFace.find(img_path=face_img, db_path=database_path, enforce_detection=False)

                    if isinstance(result, list):
                        result = result[0]

                    if not result.empty:
                        name = result.iloc[0]['identity'].split("/")[-1]
                        person_label.config(text=f"Kişi: {name}")
                    else:
                        person_label.config(text="Kişi: Tanınmıyor")

                except Exception as e:
                    print(f"Error: {e}")
                    person_label.config(text="Kişi: Hata")

            if face_detected:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))


recognition_thread = threading.Thread(target=recognize)
recognition_thread.daemon = True
recognition_thread.start()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
