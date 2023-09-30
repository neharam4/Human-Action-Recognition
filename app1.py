from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the LRCN model
model_path = r"C:\Users\manu\OneDrive\Desktop\Human Action Recognition\model_cnn_ucf101.h5"
model = load_model(model_path)

# Define the classes for classification
classes = ["WalkingWithDog", "TaiChi", "HorseRace", "Basketball", "Diving", "Drumming", "TennisSwing"]


def predict_action(image_path):
    # Load the image
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

    return class_name, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part!"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file!"
        if file:
            file_path = "uploaded_image.jpg"
            file.save(file_path)
            class_name, confidence = predict_action(file_path)
            text = "{}: {:.2f}%".format(class_name, confidence * 100)
            image = cv2.imread(file_path)
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite("static/result_image.jpg", image)
            return render_template("result.html", class_name=class_name, confidence=confidence)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
