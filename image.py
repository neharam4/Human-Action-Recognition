import cv2
import numpy as np
from keras.models import load_model
# from google.colab.patches import cv2_imshow

# Load the LRCN model
model_path = r"E:\projects\humanaction - Copy\models\model_cnn_ucf101.h5"
model = load_model(model_path)

# Define the classes for classification
classes=["WalkingWithDog", "TaiChi", "HorseRace","Basketball","Diving","Drumming","TennisSwing"]


# Load the image
image_path = r"E:\projects\humanaction - Copy\input_images\org_drum.jpg"
image = cv2.imread(image_path)

# Resize the image to 64x64 and create a list of 20 copies
resized_images = [cv2.resize(image, (64, 64)) for _ in range(20)]

# Stack the list of images along the first axis to create a tensor with shape (1, 20, 64, 64, 3)
tensor_image = np.stack(resized_images, axis=0)
tensor_image = np.expand_dims(tensor_image, axis=0)

# Make a prediction using the LRCN model
prediction = model.predict(tensor_image)[0]
class_index = np.argmax(prediction)
class_name = classes[class_index]
confidence = prediction[class_index]

# Display the predicted class and confidence
print("Class: {}, Confidence: {}".format(class_name, confidence))

# Draw the predicted class and confidence on the image
text = "{}: {:.2f}%".format(class_name, confidence*100)

cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
