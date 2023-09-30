import cv2
import numpy as np
from keras.models import load_model

# Load the LRCN model
model_path = r"D:\projects\humanaction\models\model_cnn1_ucf101.h5"
model = load_model(model_path)

# Define the classes for classification
classes = ["WalkingWithDog", "TaiChi", "HorseRace","Basketball", "Diving", "Drumming", "TennisSwing"]

# Open the video file
video_path = r"D:\projects\humanaction\input_vedios\v_Basketball_g02_c02.avi"
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to 64x64 and create a list of 20 copies
    resized_frames = [cv2.resize(frame, (64, 64)) for _ in range(20)]

    # Stack the list of frames along the first axis to create a tensor with shape (1, 20, 64, 64, 3)
    tensor_frames = np.stack(resized_frames, axis=0)
    tensor_frames = np.expand_dims(tensor_frames, axis=0)

    # Make a prediction using the LRCN model
    prediction = model.predict(tensor_frames)[0]
    class_index = np.argmax(prediction)
    class_name = classes[class_index]
    confidence = prediction[class_index]

    # Draw the predicted class and confidence on the frame
    text = "{}: {:.2f}%".format(class_name, confidence*100)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for 'q' key to exit
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
