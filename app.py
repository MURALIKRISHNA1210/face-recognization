from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# Initialize face recognition code and webcam capture

images = ['venkat sai.png', 'pramod.jpg']
known_face_encodings = [face_recognition.face_encodings(cv2.imread(image))[0] for image in images]
known_faces = [name.split('.')[0] for name in images]

cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        ret, frame = cap.read()
        resize_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        resize_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
        new_face_locations = face_recognition.face_locations(resize_frame)
        new_face_encodings = face_recognition.face_encodings(resize_frame, new_face_locations)

        for face_location, face_encoding in zip(new_face_locations, new_face_encodings):
            top, right, bottom, left = face_location
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            least_distance_index = np.argmin(face_distances)

            if matches[least_distance_index]:
                cv2.rectangle(resize_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(resize_frame, known_faces[least_distance_index], (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(resize_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(resize_frame, 'Unknown', (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        #     cv2.imshow('Face Detection', resize_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        # cap.release()
        # cv2.destroyAllWindows()
        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
