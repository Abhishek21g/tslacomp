import os
import concurrent.futures
import subprocess

# Define the directory where videos are located
videos_dir = "/home/ubuntu/tesla/tesla-real-world-video-q-a/videos/videos"

# List all videos in the directory
videos = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]

# Define the command to run YOLOv5 detection on each video
def process_video(video):
    input_video = os.path.join(videos_dir, video)
    output_dir = f"/home/ubuntu/tesla/yolov5/runs/detect/{video.split('.')[0]}"
    
    # Run the YOLOv5 detection command
    command = f"python3 /home/ubuntu/tesla/yolov5/detect.py --source {input_video} --weights /home/ubuntu/tesla/yolov5/yolov5s.pt --conf 0.4 --save-txt --save-crop --device 0 --project {output_dir}"
    
    # Run the command
    subprocess.run(command, shell=True)

# Use ThreadPoolExecutor or ProcessPoolExecutor to run the process in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_video, videos)
