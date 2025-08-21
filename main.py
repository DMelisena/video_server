
from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video file found', 400
    
    video = request.files['video']
    
    # You can optionally save the video file
    # video.save('uploaded_video.mp4')

    return 'its a video'

@app.route('/getTest', methods=['GET'])
def upload_video():
    return 'server is online'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
