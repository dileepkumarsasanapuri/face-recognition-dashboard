import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# === Configuration ===
known_faces_dir = "known_faces"
log_file_path = "recognition_log.csv"
padding = 30  # Increase box size around the face

# === Setup ===
os.makedirs(known_faces_dir, exist_ok=True)

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Setup video capture
video_capture = cv2.VideoCapture(0)

# Setup CSV log file
log_exists = os.path.isfile(log_file_path)
log_file = open(log_file_path, mode="a", newline="")
log_writer = csv.writer(log_file)

if not log_exists:
    log_writer.writerow(["Timestamp", "Name"])

print("Face Recognition Started. Press 'q' to quit, 'a' to add new faces.")

# === Main Loop ===
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

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

            # Log recognized name with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_writer.writerow([timestamp, name])
            log_file.flush()

        # Apply padding
        top, right, bottom, left = face_location
        top = max(top - padding, 0)
        right = min(right + padding, frame.shape[1])
        bottom = min(bottom + padding, frame.shape[0])
        left = max(left - padding, 0)

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q'
    if key == ord('q'):
        break

    # Add new face(s) on 'a'
    if key == ord('a'):
        print("Capturing image(s) for new face(s)...")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_locations or not face_encodings:
            print("No faces found. Try again.")
            continue

        base_name = input("Enter the base name (e.g., 'John' or 'Group1'): ")

        for idx, (loc, encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = loc
            top = max(top - padding, 0)
            right = min(right + padding, frame.shape[1])
            bottom = min(bottom + padding, frame.shape[0])
            left = max(left - padding, 0)

            face_image = frame[top:bottom, left:right]
            name = f"{base_name}_{idx+1}" if len(face_locations) > 1 else base_name
            image_path = os.path.join(known_faces_dir, f"{name}.jpg")
            cv2.imwrite(image_path, face_image)
            print(f"Saved {name} to {image_path}")

            known_face_encodings.append(encoding)
            known_face_names.append(name)

# === Cleanup ===
log_file.close()
video_capture.release()
cv2.destroyAllWindows()
print("Program exited cleanly.")
