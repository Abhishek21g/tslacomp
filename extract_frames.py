import cv2
import os

def extract_frames(video_path, output_folder, skip=10):  # 'skip' determines how many frames to skip
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_index = 0  # Keeps track of all frames, but only saves every 'skip' frame

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_index % skip == 0:  # Only save every 'skip' frame
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
            frame_index += 1
        else:
            break

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

# Example usage: Only save every 10th frame
extract_frames('tesla-real-world-video-q-a/videos/videos/00001.mp4', 'extracted_frames', skip=10)
