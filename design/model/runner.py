# smartfit.py
import cv2
import os
import glob
from modelLoader import PredictionModule
from assessment import PostureAssessment

class SmartFit:
    def __init__(self, input_dir, account_user, model_checkpoint, output_base):
        self.input_dir = input_dir
        self.account_user = account_user
        self.model_checkpoint = model_checkpoint
        self.output_base = output_base

    def get_latest_video(self):
        # Search for all .mp4 files in a directory
        video_files = glob.glob(os.path.join(self.input_dir, ".mp4"))
        if not video_files:
            raise FileNotFoundError("No videos found")
        latest_video = max(video_files, key=os.path.getmtime)
        return latest_video

    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        #width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        width  = 1280
        height = 720

        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Class modules
        pred_module = PredictionModule()
        posture_assessor = PostureAssessment()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_frames = []
        current_frame = 0
        segment_duration_sec = 2.0 # Duration of clip that the model will predict
        segment_frame_count = int(segment_duration_sec * fps)
        video_path = input_video_path  # Use the same video for clip extraction

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            segment_frames.append(frame)
            current_frame += 1

            if len(segment_frames) == segment_frame_count or current_frame == frame_count:
                segment_index = (current_frame - len(segment_frames)) / fps
                start_sec = segment_index
                end_sec = start_sec + segment_duration_sec

                predicted_action = pred_module.predict(video_path, start_sec, end_sec)
                
                rep_frame = segment_frames[len(segment_frames) // 2]
                rep_frame_rgb = cv2.cvtColor(rep_frame, cv2.COLOR_BGR2RGB)
                feedback = posture_assessor.get_landmarks(rep_frame_rgb, predicted_action)

                # Add annotations for visualization, para makita kung ano ganap
                for frm in segment_frames:
                    cv2.putText(frm, f"Action: {predicted_action}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frm, f"Feedback: {feedback}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    out.write(frm)

                segment_frames = []

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def run(self):
        latest_video_file = self.input_dir
        #latest_video_file = self.get_latest_video() # Change into this later
        
        print(f"Latest video found: {latest_video_file}")
        user_output_folder = os.path.join(self.output_base, self.account_user)
        os.makedirs(user_output_folder, exist_ok=True)
        
        video_filename = os.path.basename(latest_video_file)
        output_video_path = os.path.join(user_output_folder, f"processed_{video_filename}")

        self.process_video(latest_video_file, output_video_path)
        print(f"Processed video saved to: {output_video_path}")
