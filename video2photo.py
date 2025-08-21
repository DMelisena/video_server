import cv2
import os

def video_to_frames(video_path, output_dir):
    """
    Converts a video file into a sequence of image frames.

    Args:
        video_path (str): The path to the input video file.
        output_dir (str): The directory where the extracted frames will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    while True:
        # Read a new frame
        success, image = vidcap.read()
        if not success:
            break

        # Define the filename and save the frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, image)
        print(f"Saved {frame_filename}")
        
        frame_count += 1

    vidcap.release()
    print(f"\nFinished converting video. Total frames saved: {frame_count}")

# --- Example Usage ---
# Path to your video file
video_file = './SD_underwater/DJI120.mp4'
# Directory to save the output frames
output_folder = 'DJI120'



# Run the conversion
video_to_frames(video_file, output_folder)
