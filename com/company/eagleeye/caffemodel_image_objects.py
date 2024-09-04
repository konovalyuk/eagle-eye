import cv2
import numpy as np
import os

# Load the image
image = cv2.imread('images/image3.jpg')
#image = cv2.imread('images/image1.jpg')
 
 
# Load the pre-trained model
model_prototxt_path = 'models/caffemodel/MobileNetSSD_deploy.prototxt'
model_caffemodel_path = 'models/caffemodel/MobileNetSSD_deploy.caffemodel'
load_model = cv2.dnn.readNetFromCaffe(model_prototxt_path, model_caffemodel_path)

# Set minimal confidence threshold
minimal_confidence = 0.2

# List of categories
categories_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Generate colors for each category
np.random.seed(600000)
colors = np.random.uniform(0, 255, size=(len(categories_list), 3))

# Get image dimensions
height, width = image.shape[0], image.shape[1]

# Prepare the blob from the image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Set the input to the model
load_model.setInput(blob)

# Perform the forward pass to get the detections
detected_objects = load_model.forward()

# Initialize lists for boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Loop over the detections
for i in range(detected_objects.shape[2]):
    confidence = detected_objects[0, 0, i, 2]
    if confidence > minimal_confidence:
        category_index = int(detected_objects[0, 0, i, 1])
        upper_left_x = int(detected_objects[0, 0, i, 3] * width)
        upper_left_y = int(detected_objects[0, 0, i, 4] * height)
        lower_right_x = int(detected_objects[0, 0, i, 5] * width)
        lower_right_y = int(detected_objects[0, 0, i, 6] * height)

        boxes.append([upper_left_x, upper_left_y, lower_right_x - upper_left_x, lower_right_y - upper_left_y])
        confidences.append(float(confidence))
        class_ids.append(category_index)

# Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, minimal_confidence, 0.3)

# Loop over the indices we are keeping after NMS
for i in indices.flatten():
    box = boxes[i]
    upper_left_x, upper_left_y, width, height = box
    lower_right_x = upper_left_x + width
    lower_right_y = upper_left_y + height
    color = colors[class_ids[i]]
    cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), color, 2)
    detected_obj_text = f'{categories_list[class_ids[i]]}: {confidences[i]:.2f}'
    cv2.putText(image, detected_obj_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Display the image with detections
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
