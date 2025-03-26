from runner import SmartFit

video_path = "/home/joker/Documents/devProjects/Python/test_case/Squat.mp4"  # Path of video directory
model_path = "/home/joker/Documents/devProjects/AppDev/Sample API/model/model.ckpt"

def main():
    smartfit = SmartFit(
    input_dir=video_path,
    account_user="mema",
    model_checkpoint=model_path,
    output_base="output"
    )

    smartfit.run() # Runs the code

if __name__ == "__main__":
    main() # Calls the code, pls gumana kaaaaaaa