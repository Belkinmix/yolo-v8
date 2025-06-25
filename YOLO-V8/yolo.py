import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# --- 1. Load Pre-trained YOLOv8 Model ---
# Load the YOLOv8n (nano) model, which is small and fast.
# The model is downloaded automatically on first use.
try:
    model = YOLO("yolov8n.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Exit if the model fails to load
    exit()

# --- 2. Object Detection Function ---
# This function takes an image as input and returns the image with detections.
def detect_objects(image_input):
    # Check if the input image is None (e.g., if the user clears the input)
    if image_input is None:
        print("No input image provided.")
        return None, "Please provide an image first."

    print(f"Received image of shape: {image_input.shape} and type: {image_input.dtype}")
    
    # Run YOLOv8 inference on the image.
    # The model.predict() method handles the necessary color conversions if needed.
    results = model.predict(image_input, conf=0.4)
    # The result object contains the plotted image with bounding boxes.
    plotted_image_bgr = results[0].plot()
    return plotted_image_bgr, "Detection complete."
    print("Object detection complete. Image plotted.")

# --- 3. Create the Gradio Web Interface ---
# We use gr.Blocks() for more control over the layout.
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 📸 YOLOv8 Object Detection
        
        Welcome! This application uses the YOLOv8 model to detect objects in real-time.
        
        **Instructions:**
        1.  Either **drag and drop** an image into the input box.
        2.  Or, click **"Open Webcam"** to use your camera. Allow browser permissions, then click **"Snapshot"** to capture an image.
        3.  Click the **"Submit"** button to see the results!
        """
    )
    
    with gr.Row():
        # Input component for uploading an image or using the webcam
        input_image = gr.Image(
            type="numpy",
            label="Input Image",
            sources=["upload", "webcam"],
            height=400,
        )
        
        # Output component to display the image with detections
        output_image = gr.Image(
            type="numpy",
            label="Output Image",
            height=400
        )
        
    # A text box to show status messages
    status_text = gr.Textbox(label="Status", interactive=False)
    
    # The submit button that triggers the detection
    submit_button = gr.Button("Submit", variant="primary")
    
    # Wire the components together:
    # When the submit button is clicked, call the detect_objects function
    # with the content of input_image. The returned values will populate
    # output_image and status_text.
    submit_button.click(
        fn=detect_objects,
        inputs=[input_image],
        outputs=[output_image, status_text]
    )

# --- 4. Launch the Application ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    # The launch() method creates a local web server.
    iface.launch(debug=True) # Set debug=True for detailed error messages in console