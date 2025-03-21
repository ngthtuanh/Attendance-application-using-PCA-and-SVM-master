import os
import cv2
from app.face_recognition import face_recognitionPipeline
from flask import render_template, request, Response, jsonify
import matplotlib.image as mating
from flask_socketio import SocketIO
from datetime import datetime
import json

UPLOAD_FOLDER = 'static/upload'
ATTENDANCE_FILE = 'attendance_records.json'

# Initialize SocketIO
socketio = SocketIO()

def init_app(app):
    socketio.init_app(app)
    return socketio

def load_attendance_records():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_attendance_records(records):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(records, f, indent=4)

def record_attendance(name, confidence):
    records = load_attendance_records()
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    if current_date not in records:
        records[current_date] = {}
    
    if name not in records[current_date]:
        records[current_date][name] = {
            'first_seen': current_time,
            'last_seen': current_time,
            'confidence': confidence
        }
    else:
        records[current_date][name]['last_seen'] = current_time
        records[current_date][name]['confidence'] = max(
            confidence, 
            records[current_date][name]['confidence']
        )
    
    save_attendance_records(records)
    return records[current_date][name]

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, predictions = face_recognitionPipeline(frame, path=False)
            
            # Process predictions and record attendance
            attendance_status = None
            if predictions:
                for pred in predictions:
                    if pred['score'] >= 0.7:
                        name = pred['prediction_name']
                        score = pred['score']
                        attendance_record = record_attendance(name, score)
                        
                        attendance_status = {
                            'name': name,
                            'score': score,
                            'first_seen': attendance_record['first_seen'],
                            'last_seen': attendance_record['last_seen']
                        }
                        
                        # Emit attendance update via SocketIO
                        socketio.emit('attendance_update', attendance_status)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def get_attendance_report():
    records = load_attendance_records()
    return jsonify(records)

def face_recognition_App():
    if request.method == 'POST':
        if 'image_name' in request.files:
            f = request.files['image_name']
            filename = f.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(path)
            pred_image, predictions = face_recognitionPipeline(path, path=True)
            pred_filename = 'prediction_image.jpg'
            cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
            
            report = []
            for i, obj in enumerate(predictions):
                gray_image = obj['roi']
                eigen_image = obj['eig_img'].reshape(100,100)
                pred_name = obj['prediction_name']
                score = round(obj['score']*100,2)
                
                # Record attendance for uploaded images as well
                if score >= 70:  # 70% confidence threshold
                    attendance_record = record_attendance(pred_name, score/100)
                
                gray_image_name = f'roi_{i}.jpg'
                eig_image_name = f'eig_{i}.jpg'
                mating.imsave(f'./static/predict/{gray_image_name}', gray_image, cmap='gray')
                mating.imsave(f'./static/predict/{eig_image_name}', eigen_image, cmap='gray')
                
                report.append([gray_image_name, eig_image_name, pred_name, score])
                
            return render_template('name.html', fileupload=True, report=report)
            
    return render_template('name.html', fileupload=False)

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')