import concurrent.futures
import cv2

def process_video(video_path):
    # Frame extraction or YOLO detection here
    print(f"Processing {video_path}...")

def process_videos(video_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_video, video_paths)

# Example usage
video_paths = ["00001.mp4", "00002.mp4", "00003.mp4"]
process_videos(video_paths)
