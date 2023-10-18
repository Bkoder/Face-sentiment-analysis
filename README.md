# Face-sentiment-analysis

Description:
This project aims to detect emotions from facial expressions using deep learning. The emotion detection model is trained on a dataset containing images of faces expressing different emotions, including Happy, Sad, Normal, Surprise, and Angry.

Files:

Training Script (train_emotion_model.py):

Uses Keras and TensorFlow for building and training the emotion detection model.
Data augmentation techniques are applied to generate variations in the training dataset.
The model architecture consists of convolutional neural network (CNN) layers for feature extraction and classification.
Model training details, callbacks, and optimization settings are included.
User Interface Script (emotion_detection_ui.py):

Utilizes OpenCV for capturing video frames from a webcam.
Haar Cascade Classifier is used for face detection in each frame.
The pre-trained emotion detection model is loaded and applied to predict the emotion from the detected face.
Results are displayed in real-time with bounding boxes and emotion labels.
Model Checkpoint (Emotion_little_vgg.h5):

Contains the saved weights of the trained emotion detection model.
Haar Cascade File (haarcascade_frontalface_default.xml):

XML file for the Haar Cascade Classifier used in face detection.
Usage:

Run train_emotion_model.py to train the emotion detection model. Adjust file paths if necessary.
Once the model is trained, run emotion_detection_ui.py to start the real-time emotion detection using your webcam.
Dependencies:

Python 3
Keras
TensorFlow
OpenCV
NumPy
Note:

Make sure to have the required Python packages installed before running the scripts.
The training dataset path and other configurations can be customized in the training script.
Ensure that the file paths for the model checkpoint and Haar Cascade classifier are correctly specified.
