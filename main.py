from flask import Flask, request, jsonify
import cv2
import os
import uuid
from datetime import datetime
import tempfile
import subprocess
import shutil
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import zipfile
from pathlib import Path

app = Flask(__name__)

# Configuration for Railway deployment
app.config['MAX_CONTENT_LENGTH'] = 500 *1024 * 1024 * 1024  # 500MB max file size

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'YOLO Fish Detection API',
        'version': '1.0.0'
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    try:
        # Create necessary directories
        os.makedirs('./yolo_crop_clustering', exist_ok=True)
        os.makedirs('./processed_video_result', exist_ok=True)
        
        # Save uploaded video with secure filename
        filename = secure_filename(video.filename)
        video_path = os.path.join('./yolo_crop_clustering', filename)
        video.save(video_path)
        
        # Check if model file exists
        model_path = './best-e100-random-fishes.pt'
        if not os.path.exists(model_path):
            # Try to find the model in the mlpackage directory
            mlpackage_path = './best-e100-random-fishes.mlpackage'
            if os.path.exists(mlpackage_path):
                return jsonify({
                    'error': 'Model is in .mlpackage format. Please provide a .pt model file for YOLO inference.'
                }), 400
            else:
                return jsonify({
                    'error': 'YOLO model file not found. Please ensure best-e100-random-fishes.pt is in the root directory.'
                }), 400
        
        # Run YOLO processing command
        cmd = [
            'python', 'yolo_crop_cluster.py',
            '--model', model_path,
            '--source', './yolo_crop_clustering',
            '--output', './processed_video_result',
            '--conf', '0.25',
            '--cluster-method', 'dbscan',
            '--save-features'
        ]
        
        # Execute the command with increased timeout for large files
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=6000)  # 10 minutes
        
        if result.returncode != 0:
            return jsonify({
                'error': 'Processing failed',
                'stderr': result.stderr,
                'stdout': result.stdout
            }), 500
        
        # Check if results exist
        result_dir = Path('./processed_video_result')
        if not result_dir.exists() or not any(result_dir.iterdir()):
            return jsonify({'error': 'No processed results found'}), 500
        
        # Create a zip file of the results
        zip_filename = f'processed_results_{uuid.uuid4().hex[:8]}.zip'
        zip_path = os.path.join('./processed_video_result/', zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Define image extensions
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            
            # Add files from clusters directory
            clusters_path = './processed_video_result/clusters/'
            if os.path.exists(clusters_path):
                for root, dirs, files in os.walk(clusters_path):
                    for file in files:
                        if file.lower().endswith(image_extensions):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, clusters_path)
                            zipf.write(file_path, arcname)
            
            # Add files from analysis directory
            analysis_path = './processed_video_result/analysis/'
            if os.path.exists(analysis_path):
                for root, dirs, files in os.walk(analysis_path):
                    for file in files:
                        if file.lower().endswith(image_extensions):
                            file_path = os.path.join(root, file)
                            arcname = os.path.join('analysis', os.path.relpath(file_path, analysis_path))
                            zipf.write(file_path, arcname)
        
        # Clean up image files after zip creation
        def cleanup_images_in_directory(directory_path):
            """Remove all image files from the specified directory and its subdirectories"""
            if os.path.exists(directory_path):
                image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
                for root, dirs, files in os.walk(directory_path, topdown=False):
                    for file in files:
                        if file.lower().endswith(image_extensions):
                            try:
                                file_path = os.path.join(root, file)
                                os.remove(file_path)
                                print(f"Deleted: {file_path}")
                            except Exception as e:
                                print(f"Warning: Could not delete {file_path}: {e}")
                    
                    # Remove empty directories
                    try:
                        if not os.listdir(root):  # Directory is empty
                            os.rmdir(root)
                            print(f"Removed empty directory: {root}")
                    except Exception as e:
                        print(f"Warning: Could not remove directory {root}: {e}")
        
        # Clean up the directories after zip creation
        cleanup_directories = [
            './processed_video_result/analysis/',
            './processed_video_result/clusters/',
            './processed_video_result/crop/'  # This will be cleaned as part of analysis/
        ]
        
        for directory in cleanup_directories:
            if os.path.exists(directory):
                cleanup_images_in_directory(directory)
                print(f"Cleaned up images in: {directory}")
        
        return jsonify({
            'message': 'Video processed successfully',
            'download_url': f'https://prime-whole-fish.ngrok-free.app/download/{zip_filename}',
            'processing_time': 'completed'
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Processing timeout (exceeded 10 minutes)'}), 500
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    finally:
        # Clean up uploaded video file
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Warning: Could not remove {video_path}: {e}")
@app.route('/download/<filename>')
def download_file(filename):
    """Route to download processed results"""
    try:
        return send_from_directory('./processed_video_result', filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/results')
def list_results():
    """Route to list all files in processed_video_result directory"""
    try:
        result_dir = Path('./processed_video_result')
        if not result_dir.exists():
            return jsonify({'files': []})
        
        files = []
        for file_path in result_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(result_dir)
                files.append({
                    'name': file_path.name,
                    'path': str(relative_path),
                    'size': file_path.stat().st_size,
                    'download_url': f'/download/{file_path.name}'
                })
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

if __name__ == '__main__':
    # Create required directories at startup
    directories = [
        'frames', 
        'yolo_crop_clustering', 
        'processed_video_result',
        'static',
        'templates'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)


# ... your existing code ...
# Now, this line should work without errors
model = YOLO(args.model)
