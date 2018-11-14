"""
Template for image recognition using the ResNet 50 model
Add the images to the working directory and set the path variable
Output is a list of labels with a probability for each image
"""
import numpy as np
import os.path
from keras.preprocessing import image
from keras.applications import resnet50

# Set the path and list available image extensions
path = "/home/kacper/Desktop/Machine Learning/Exercise Files/Lynda/05"
image_extensions = ["jpg", "png", "gif", "jpeg"]

# Load all of the Images from the workspace folder
image_list = []
for extension in image_extensions:
    for image_dir  in os.listdir(path):
        if extension in image_dir:
            image_list.append(image_dir)

# Convert images in 3D matrices
x   = []
for image_dir in image_list:
    img = image.load_img(image_dir, target_size = (224, 224))
    x.append(image.img_to_array(img))
x = np.array(x)    

# Resnet50 with ImageNet weights
model = resnet50.ResNet50()

# Preprocess the images
x = resnet50.preprocess_input(x)

# Predict 
predictions = model.predict(x)

# Extract the results
predicted_classes = resnet50.decode_predictions(predictions, top=3)
for i, predicted_class in enumerate(predicted_classes):
    print("Image: " + image_list[i])
    for imagenet_id, name, likelihood in predicted_class:
        print(" - {}: {:0.3f}%".format(name, likelihood*100))

