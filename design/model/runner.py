import os
import glob
import cv2
import imageio
import random
from modelLoader import PredictionModule
from assessment import PostureAssessment

class SmartFit:
    def __init__(self, input_dir, account_user, model_checkpoint, output_base):
        self.input_dir = input_dir
        self.account_user = account_user
        self.model_checkpoint = model_checkpoint
        self.output_base = output_base

        self.file_name = list()

    def get_latest_video(self):
        # Search for all .mp4 files in the directory
        video_files = glob.glob(os.path.join(self.input_dir, "*.mp4"))
        if not video_files:
            raise FileNotFoundError("No videos found")
        latest_video = max(video_files, key=os.path.getmtime)
        return latest_video

    def process_video(self, input_video_path, output_video_path):
        reader = imageio.get_reader(input_video_path)
        meta_data = reader.get_meta_data()
        fps = meta_data['fps']
        writer = imageio.get_writer(output_video_path, fps=fps)

        # Initialize class modules
        pred_module = PredictionModule()
        posture_assessor = PostureAssessment()

        pred_arr = []
        segment_frames = []
        segment_duration_sec = 2.0  # Duration for each clip, 2 for good measure na lang
        segment_frame_count = int(segment_duration_sec * fps)
        current_frame = 0

        # Process video frame-by-frame using imageio
        for frame in reader:
            segment_frames.append(frame)
            current_frame += 1

            if len(segment_frames) == segment_frame_count:
                start_sec = (current_frame - len(segment_frames)) / fps
                end_sec = start_sec + segment_duration_sec

                # Predict action based on the current segment
                predicted_action = pred_module.predict(input_video_path, start_sec, end_sec)
                pred_arr.append(predicted_action) # Append predictions

                # Use the middle frame for posture assessment
                rep_frame = segment_frames[len(segment_frames) // 2]
                feedback = posture_assessor.get_landmarks(rep_frame, predicted_action)
                

                # Annotate each frame in the segment with action and feedback text
                for frm in segment_frames:
                    frm_bgr = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
                    cv2.putText(frm_bgr, f"Action: {predicted_action}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(frm_bgr, f"Feedback: {feedback}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    annotated_frame = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2RGB)
                    writer.append_data(annotated_frame)
                
                segment_frames = []
        
        # Check for leftover frames
        if segment_frames:
            start_sec = (current_frame - len(segment_frames)) / fps
            end_sec = current_frame / fps  # Duration might be less than segment_duration_sec
            predicted_action = pred_module.predict(input_video_path, start_sec, end_sec)
            rep_frame = segment_frames[len(segment_frames) // 2]
            feedback = posture_assessor.get_landmarks(rep_frame, predicted_action)

            for frm in segment_frames:
                frm_bgr = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
                cv2.putText(frm_bgr, f"Action: {predicted_action}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frm_bgr, f"Feedback: {feedback}", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                annotated_frame = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2RGB)
                writer.append_data(annotated_frame)

        writer.close()
        reader.close()

        #print(pred_arr)
        self.file_name = pred_arr
    
    def renameFile(self, arr):
        return max(set(arr), key=arr.count)

    def run(self):
        random.seed(0)
        latest_video_file = self.get_latest_video()
        #print(f"Latest video found: {latest_video_file}")

        user_output_folder = os.path.join(self.output_base, self.account_user)
        os.makedirs(user_output_folder, exist_ok=True)
        video_filename = os.path.basename(latest_video_file)
        
        temp_output_video_path = os.path.join(user_output_folder, f"temp_file_{video_filename}")

        # For a much more cleaner removal, para walang error
        try:
            self.process_video(latest_video_file, temp_output_video_path)
            most_predicted = self.renameFile(self.file_name)
            new_output_video_path = os.path.join(user_output_folder, f"{random.randint(0,9999)}_{most_predicted}.mp4")

            os.rename(temp_output_video_path, new_output_video_path)
            #print(f"Processed video saved to: {new_output_video_path}")
        finally:
            if os.path.exists(temp_output_video_path):
                os.remove(temp_output_video_path) # Remove temp

