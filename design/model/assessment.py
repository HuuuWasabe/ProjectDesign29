import cv2
import mediapipe as mp
import numpy as np
import os

class Posture:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # These parameters (min_detection_confidence and min_tracking_confidence)
        # can be adjusted; ISO 11228‑3 recommends safe postures for manual handling tasks.
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Middle point
        c = np.array(c)  # Last point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return angle
    
    def assess_barbell_curl(self, landmarks):
        # Note: ISO standards for manual handling do not provide specific elbow ranges for exercise.
        # The following range (50° to 150°) is an approximation to ensure a controlled motion.
        left_elbow_angle = self.calculate_angle(
            [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y],
        )
        if 50 <= left_elbow_angle <= 150:
            return "Correct posture!"
        return "Incorrect elbow angle. Keep the motion controlled."

    def assess_deadlift(self, landmarks):
        # ISO 11228‑3 recommends that, during manual handling, the back (spine) should be kept relatively straight.
        # Here we check that the back (angle between shoulder, hip, and knee) is between 160° and 180°.
        back_angle = self.calculate_angle(
            [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        )
        if 160 <= back_angle <= 180:
            return "Correct posture!"
        return "Incorrect back alignment. Maintain a straight back."

    def assess_squat(self, landmarks):
        # For squats (a common manual handling movement), ISO-related guidelines favor avoiding extreme flexion.
        # Here, an acceptable knee angle is set between 90° and 120°.
        knee_angle = self.calculate_angle(
            [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        )
        if 90 <= knee_angle <= 120:
            return "Correct posture!"
        return "Incorrect squat depth. Ensure proper leg joint angle."

    def assess_lateral_raises(self, landmarks):
        # ISO 11228‑3 guidelines (when adapted for safe shoulder postures) advise keeping arm elevation
        # within a safe range (typically below 80° for prolonged work). For lateral raises, however, you are
        # intentionally lifting your arms roughly parallel to the floor (about 90°). Here we allow a range around 90°.
        shoulder_angle = self.calculate_angle(
            [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y],
        )
        if 80 <= shoulder_angle <= 100:
            return "Correct posture!"
        return "Incorrect shoulder angle. Aim to lift arms parallel to the floor."

    def assess_overhead_press(self, landmarks):
        # For overhead press, keeping the elbows nearly fully extended (close to 180°) is recommended.
        # Here, we check that the elbow angle (wrist-elbow-shoulder) is between 160° and 180°.
        elbow_angle = self.calculate_angle(
            [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        )
        if 160 <= elbow_angle <= 180:
            return "Correct posture!"
        return "Incorrect elbow form. Fully extend arms overhead."

class PostureAssessment(Posture):
    def __init__(self):
        super().__init__()
        self.feedback = ""

    def get_landmarks(self, image, exercise):
        try:
            results = self.pose.process(image)
            landmarks = results.pose_landmarks.landmark
                
            if exercise == "BarbellCurl":
                self.feedback = self.assess_barbell_curl(landmarks)
            elif exercise == "Deadlift":
                self.feedback = self.assess_deadlift(landmarks)
            elif exercise == "Squat":
                self.feedback = self.assess_squat(landmarks)
            elif exercise == "LateralRaises":
                self.feedback = self.assess_lateral_raises(landmarks)
            elif exercise == "OverheadPress":
                self.feedback = self.assess_overhead_press(landmarks)
        
        except Exception as e:
            self.feedback = "Assessing..."
        return self.feedback
