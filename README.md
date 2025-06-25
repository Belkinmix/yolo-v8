# 🧠 YOLOv8 Object Detection App

This is a simple object detection app built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Gradio](https://gradio.app/).  
It allows users to upload images or use their webcam to detect objects in real-time with bounding boxes.

## 🚀 Features

- 📷 Upload an image or capture a photo using your webcam
- 🧠 Detects objects using YOLOv8 with bounding boxes and labels
- 🌐 Clean web-based interface using Gradio
- ⚙️ Adjustable detection confidence threshold (via code)

## 🧾 How to Use

After running the app:
Open the URL printed in your terminal — usually something like: http://127.0.0.1:7860

⚠️ This app runs locally. You must open it from the terminal output in your browser.

## 🧠 Model Info
This app uses the yolov8n.pt (YOLOv8 Nano) model.
If it's not found locally, Ultralytics will download it automatically on first use.

## 💻 Requirements
Python 3.8+
Gradio
Ultralytics (YOLOv8)
OpenCV
NumPy

Made by Mikhail Belkin
