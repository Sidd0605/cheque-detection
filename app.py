# from flask import Flask, render_template, request, redirect, url_for
# import torch
# from ultralytics import YOLO
# import os
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import random

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = '/Users/siddhanthsalian/Desktop/miscot /cheque deploy/static/uploads'
# app.config['RESULT_FOLDER'] = '/Users/siddhanthsalian/Desktop/miscot /cheque deploy/static/results'

# # Load the YOLOv8 model
# model = YOLO('/Users/siddhanthsalian/Desktop/miscot /cheque deploy/best.pt')

# # Class names as per your defined order
# class_names = {
#     0: 'Date',
#     1: 'Amount in word',
#     2: 'Name',
#     3: 'Account No',
#     4: 'Security Code',
#     5: 'Cheque No',
#     6: 'Transaction Code',
#     7: 'MICR',
#     8: 'Amount in figures',
#     9: 'IFSC'
# }

# # Function to process the uploaded image and save the result
# def process_image(image_path, output_folder):
#     # Load an image
#     img = Image.open(image_path)

#     # Run YOLOv8 model on the image
#     results = model(image_path)

#     # Get the detection results (bounding boxes and class IDs)
#     detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
#     class_ids = results[0].boxes.cls.cpu().numpy()    # Class IDs

#     # Dictionary to store cropped images by class ID
#     cropped_images_by_class = {i: [] for i in range(10)}  # Prepares dictionary for each class

#     # Process each detection and crop the image
#     for idx, (x_min, y_min, x_max, y_max, class_id) in enumerate(zip(detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3], class_ids)):
#         # Convert coordinates to integer values for cropping
#         x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

#         # Crop the detected part of the image
#         cropped_img = img.crop((x_min, y_min, x_max, y_max))

#         # Append the cropped image to the dictionary under its class ID
#         cropped_images_by_class[int(class_id)].append(cropped_img)

#     # Concatenate cropped images in the specified class order
#     ordered_cropped_images = []
#     for class_id in range(10):
#         # Extend the list with all cropped images corresponding to the current class ID
#         ordered_cropped_images.extend(cropped_images_by_class[class_id])

#     # If there are any cropped images, concatenate them vertically
#     if ordered_cropped_images:
#         # Find the maximum width and total height to stack the images vertically
#         widths, heights = zip(*(i.size for i in ordered_cropped_images))
#         total_height = sum(heights)
#         max_width = max(widths)

#         # Create a blank canvas to paste the cropped images
#         combined_img = Image.new('RGB', (max_width, total_height))

#         # Paste each cropped image onto the canvas in the specified order
#         y_offset = 0
#         for cropped_img in ordered_cropped_images:
#             combined_img.paste(cropped_img, (0, y_offset))
#             y_offset += cropped_img.height

#         # Generate a unique filename using timestamp and random number
#         timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
#         random_suffix = random.randint(1000, 9999)  # Adds a random number for uniqueness
#         image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the base name of the image
#         save_path = os.path.join(output_folder, f'{image_name}_{timestamp}_{random_suffix}.jpg')
        
#         # Save the final combined image
#         combined_img.save(save_path)
#         # return save_path
#         return f'/static/results/{os.path.basename(save_path)}'
#     else:
#         return None

# # Route for the home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route to handle image upload and display results
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return redirect(request.url)

#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Process the uploaded image and save the result
#         result_image_path = process_image(filepath, app.config['RESULT_FOLDER'])

#         if result_image_path:
#             return render_template('index.html', result_image=result_image_path)
#         else:
#             return render_template('index.html', error="No objects detected")

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import torch
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
# import time
# import random

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Load the YOLOv8 model
model = YOLO('best.pt')

# Class names as per your defined order
class_names = {
    0: 'Date',
    1: 'Amount in word',
    2: 'Name',
    3: 'Account No',
    4: 'Security Code',
    5: 'Cheque No',
    6: 'Transaction Code',
    7: 'MICR',
    8: 'Amount in figures',
    9: 'IFSC'
}

# Function to process the uploaded image and save the result
def process_image(image_path, output_folder):
    # Load an image
    img = Image.open(image_path)

    # Run YOLOv8 model on the image
    results = model(image_path)

    # Get the detection results (bounding boxes and class IDs)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    class_ids = results[0].boxes.cls.cpu().numpy()    # Class IDs

    # Dictionary to store cropped images by class ID
    cropped_images_by_class = {i: [] for i in range(10)}  # Prepares dictionary for each class

    # Process each detection and crop the image
    for idx, (x_min, y_min, x_max, y_max, class_id) in enumerate(zip(detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3], class_ids)):
        # Convert coordinates to integer values for cropping
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Crop the detected part of the image
        cropped_img = img.crop((x_min, y_min, x_max, y_max))

        # Append the cropped image to the dictionary under its class ID
        cropped_images_by_class[int(class_id)].append(cropped_img)

    # Concatenate cropped images in the specified class order
    ordered_cropped_images = []
    for class_id in range(10):
        # Extend the list with all cropped images corresponding to the current class ID
        ordered_cropped_images.extend(cropped_images_by_class[class_id])

    # If there are any cropped images, concatenate them vertically
    if ordered_cropped_images:
        # Find the maximum width and total height to stack the images vertically
        widths, heights = zip(*(i.size for i in ordered_cropped_images))
        total_height = sum(heights)
        max_width = max(widths)

        # Create a blank canvas to paste the cropped images
        combined_img = Image.new('RGB', (max_width, total_height))

        # Paste each cropped image onto the canvas in the specified order
        y_offset = 0
        for cropped_img in ordered_cropped_images:
            combined_img.paste(cropped_img, (0, y_offset))
            y_offset += cropped_img.height

        # Generate a filename using the original image name
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the base name of the image
        save_path = os.path.join(output_folder, f'{image_name}_output.jpg')
        
        # Save the final combined image
        combined_img.save(save_path)
        return f'/static/results/{os.path.basename(save_path)}'
    else:
        return None


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and display results
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded image and save the result
        result_image_path = process_image(filepath, app.config['RESULT_FOLDER'])

        if result_image_path:
            # Return both the input image path and result image path
            return render_template('index.html', input_image=f'/static/uploads/{filename}', result_image=result_image_path)
        else:
            return render_template('index.html', error="No objects detected")


if __name__ == '__main__':
    app.run(debug=True)
