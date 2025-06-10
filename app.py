from flask import Flask, render_template, Response, request, redirect
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import csv
from flask import jsonify
from flask import request, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
known_faces_dir = 'known_faces'
log_file_path = 'recognition_log.csv'
os.makedirs(known_faces_dir, exist_ok=True)

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

# Ensure recognition log exists
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Name'])

# Initialize webcam
video_capture = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index]

                    # Log recognized face
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, name])

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_face', methods=['POST'])
def add_face():
    name = request.form['name']
    success, frame = video_capture.read()
    if success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        if face_locations and face_encodings:
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]
            filename = os.path.join(known_faces_dir, f"{name}.jpg")
            cv2.imwrite(filename, face_image)

            load_known_faces()

    return redirect('/')

@app.route('/logs')
def logs():
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    return '<br>'.join(lines)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard_data')
def dashboard_data():
    from collections import Counter

    with open(log_file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    entries = [line.strip().split(",") for line in lines if line.strip()]
    total = len(entries)

    counter = Counter(name for _, name in entries)
    names = list(counter.keys())
    counts = list(counter.values())

    recent = [{"timestamp": ts, "name": name} for ts, name in entries[-10:]]

    return jsonify({
        "total": total,
        "names": names,
        "counts": counts,
        "recent": recent[::-1]  # Show newest first
    })

UPLOAD_FOLDER = 'known_faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max 5MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_face():
    if 'file' not in request.files or 'name' not in request.form:
        return redirect('/')

    file = request.files['file']
    name = request.form['name'].strip().replace(" ", "_")

    if file.filename == '' or not allowed_file(file.filename):
        return redirect('/')

    filename = secure_filename(f"{name}.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Encode and add to known faces
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

    return redirect('/')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
