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
        
        # Run YOLO processing command
        cmd = [
            'python', 'yolo_crop_cluster.py',
            '--model', './best-e100-random-fishes.pt',
            '--source', './yolo_crop_clustering',
            '--output', './processed_video_result',
            '--conf', '0.25',
            '--cluster-method', 'dbscan',
            '--save-feature'
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
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
        zip_filename = 'processed_results.zip'
        zip_path = os.path.join('./processed_video_result', zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk('./processed_video_result'):
                for file in files:
                    if file != zip_filename:  # Don't include the zip file itself
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, './processed_video_result')
                        zipf.write(file_path, arcname)
        
        return jsonify({
            'message': 'Video processed successfully',
            'download_url': f'/download/{zip_filename}',
            'stdout': result.stdout
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Processing timeout (exceeded 5 minutes)'}), 500
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    finally:
        # Clean up uploaded video file
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)

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
    # Create frames directory if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    app.run(host='0.0.0.0', port=8081, debug=True)
