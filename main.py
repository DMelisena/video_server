from flask import Flask, request, jsonify
import cv2
import os
import uuid
from datetime import datetime
import tempfile

app = Flask(__name__)

def get_video_creation_date(temp_path):
    """
    Extract video creation date from metadata
    Returns the creation date or None if not available
    """
    try:
        # Try to get file creation time from filesystem
        creation_time = os.path.getctime(temp_path)
        return datetime.fromtimestamp(creation_time).isoformat()
    except:
        try:
            # Alternative: try to get from video metadata using OpenCV
            vidcap = cv2.VideoCapture(temp_path)
            # Note: OpenCV doesn't provide direct metadata access
            # This is a fallback that returns None
            vidcap.release()
            return None
        except:
            return None

def video_to_frames_from_memory(video_data, output_dir, target_fps=5):
    """
    Converts video data (from memory) into a sequence of image frames at a specific frame rate.
    Args:
        video_data (bytes): The video file data in memory
        output_dir (str): The directory where the extracted frames will be saved.
        target_fps (int): Target frames per second to extract (default: 5)
    Returns:
        dict: Information about the extraction process including video creation date
    """
    import tempfile
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a temporary file to work with OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_data)
        temp_path = temp_file.name
    
    # Get video creation date before processing
    video_creation_date = get_video_creation_date(temp_path)
    
    try:
        # Open the video file
        vidcap = cv2.VideoCapture(temp_path)
        if not vidcap.isOpened():
            return {
                'success': False,
                'error': 'Could not process video data',
                'frames_saved': 0
            }
        
        # Get video properties
        original_fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Calculate frame interval to achieve target fps
        frame_interval = original_fps / target_fps if original_fps > 0 else 1
        
        frame_count = 0
        saved_count = 0
        
        while True:
            # Read a new frame
            success, image = vidcap.read()
            if not success:
                break
            
            # Check if this frame should be saved based on the interval
            if frame_count % int(frame_interval) == 0:
                # Calculate timestamp for filename
                timestamp = frame_count / original_fps if original_fps > 0 else frame_count
                
                # Define the filename and save the frame
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}_t{timestamp:.2f}s.jpg")
                cv2.imwrite(frame_filename, image)
                saved_count += 1
            
            frame_count += 1
        
        vidcap.release()
        
        actual_fps = saved_count / duration if duration > 0 else 0
        
        return {
            'success': True,
            'original_fps': original_fps,
            'duration': duration,
            'total_frames_processed': frame_count,
            'frames_saved': saved_count,
            'actual_extraction_fps': actual_fps,
            'output_directory': output_dir,
            'video_creation_date': video_creation_date
        }
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found'}), 400
    
    video = request.files['video']
    
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    try:
        # Generate unique identifier for this upload
        upload_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processing_date = datetime.now().isoformat()
        
        # Create frames directory (no need to save video)
        frames_dir = f"frames/{timestamp}_{upload_id}"
        
        # Read video data directly into memory
        video_data = video.read()
        
        # Extract frames directly from memory
        extraction_result = video_to_frames_from_memory(video_data, frames_dir, target_fps=5)
        
        if extraction_result['success']:
            response_data = {
                'message': 'Frames extracted successfully from uploaded video',
                'upload_id': upload_id,
                'original_video_name': video.filename,
                'frames_directory': frames_dir,
                'dates': {
                    'video_creation_date': extraction_result.get('video_creation_date'),
                    'processing_date': processing_date
                },
                'extraction_info': {
                    'original_fps': round(extraction_result['original_fps'], 2),
                    'duration_seconds': round(extraction_result['duration'], 2),
                    'frames_extracted': extraction_result['frames_saved'],
                    'extraction_fps': round(extraction_result['actual_extraction_fps'], 2)
                }
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                'error': 'Failed to extract frames',
                'details': extraction_result['error']
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Optional: Route to list frame folders
@app.route('/frames', methods=['GET'])
def list_frames():
    if not os.path.exists('frames'):
        return jsonify({'frame_folders': []})
    
    frame_folders = []
    for folder in os.listdir('frames'):
        folder_path = os.path.join('frames', folder)
        if os.path.isdir(folder_path):
            frame_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            
            frame_folders.append({
                'folder': folder,
                'frames_extracted': frame_count,
                'frames_directory': folder_path
            })
    
    return jsonify({'frame_folders': frame_folders})

if __name__ == '__main__':
    # Create frames directory if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    app.run(host='0.0.0.0', port=8081, debug=True)
