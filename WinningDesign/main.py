import cv2
import mediapipe as mp
import numpy as np
import os

import keras
from keras import layers


@keras.saving.register_keras_serializable()
class SEBlock(keras.layers.Layer):
    def __init__(self, filters, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.global_avg_pool = layers.GlobalAveragePooling3D()
        self.fc1 = layers.Dense(filters // reduction, activation='relu')
        self.fc2 = layers.Dense(filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, 1, filters))

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return inputs * se

@keras.saving.register_keras_serializable()
class X3DBottleneck(keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1, 1), **kwargs):
        super(X3DBottleneck, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides

        # Define the layers inside the bottleneck block
        self.conv1 = layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=strides, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")
        self.bn3 = layers.BatchNormalization()

        # Shortcut layer for residual connection
        self.shortcut = layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=strides, padding="same")
        self.shortcut_bn = layers.BatchNormalization()

        self.add = layers.Add()
        self.relu_out = layers.ReLU()

    def call(self, inputs, training=False):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Shortcut path
        shortcut = self.shortcut(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        # Add main and shortcut paths
        x = self.add([x, shortcut])
        x = self.relu_out(x)

        return x

## Load model
# Path to the model
model_path = '/mnt/d/vscodeProjects/Python/model_X3D.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Attempt to load the model
try:
    load_model = keras.models.load_model(
        model_path,
        custom_objects={'SEBlock': SEBlock, 'X3DBottleneck': X3DBottleneck}
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

## Define the dimensions of frames in the set of frames created (Default params muna)
HEIGHT = 224
WIDTH = 224
SEQUENCE_LENGTH = 24 # small frame raw para small memory at computation ang gawin, ndi kakakayanin ng mga nasa 16GB lang
LABELS = sorted(['BarbellCurl', 'Deadlift', 'Squat', 'LateralRaises', 'OverheadPress', 'Standing', 'Walking', 'Sitting'
          ]) # Eto muna

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

sequence_length = SEQUENCE_LENGTH
resize_shape = (HEIGHT, WIDTH)
file_dir = 'unknown/Squat.MOV'

cap = cv2.VideoCapture(file_dir)
frames = []
predicted_label = ""
pred_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Create a copy of the original frame before any modifications
    raw_frame = frame.copy()

    # Process the frame for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw keypoints and connections on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                  )

    # Resize and prepare frame for prediction
    resized_frame = cv2.resize(frame, resize_shape)
    rgb_resized_frame = resized_frame[:, :, ::-1]
    frames.append(rgb_resized_frame)

    # Add predicted activity label to frame
    display_text = f"Activity: {predicted_label}"
    cv2.putText(frame, display_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.1, (0, 255, 0), 5, cv2.LINE_AA)

    # Stack the raw frame and processed frame horizontally
    combined = np.hstack((raw_frame, frame))
    cv2.imshow('Cam1', combined)

    # Perform prediction for every frames
    """
    if len(frames) == sequence_length:
        frames_array = np.array(frames) / 255.0  # Normalize to [0, 1]
        frames_array = np.expand_dims(frames_array, axis=0)  # Shape (1, sequence_length, height, width, 3)

        prediction = load_model.predict(frames_array)

        # Update predicted label
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = sorted(LABELS)[predicted_index]
        print("Activity:", predicted_label)

        frames = [] """

    frames_array = np.array(frames) / 255.0  # Normalize to [0, 1]
    frames_array = np.expand_dims(frames_array, axis=0)  # Shape (1, sequence_length, height, width, 3)

    prediction = load_model.predict(frames_array)

    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = sorted(LABELS)[predicted_index]
    print("Activity:", predicted_label)

    pred_list.append(predicted_label)

    frames = []

    # Exit loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(pred_list)