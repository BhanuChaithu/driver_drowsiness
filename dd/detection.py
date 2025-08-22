import cv2
import time
import pygame
import mediapipe as mp
import threading
import tkinter as tk

# ----------------- Initialize Pygame for Alarm -----------------
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Ensure alarm.wav is in the same folder

# ----------------- Initialize MediaPipe Face Mesh -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ----------------- Parameters -----------------
EYE_CLOSED_THRESHOLD = 2.0  # seconds
eye_closed_start = None
alarm_triggered = False
running = False  # Flag to control loop

# ----------------- Eye Landmark Indices -----------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ----------------- EAR Calculation Function -----------------
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[5]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[4]]
    p5 = landmarks[eye_indices[0]]
    p6 = landmarks[eye_indices[3]]

    vertical1 = ((p2.x - p4.x)**2 + (p2.y - p4.y)**2)**0.5
    vertical2 = ((p3.x - p5.x)**2 + (p3.y - p5.y)**2)**0.5
    horizontal = ((p1.x - p6.x)**2 + (p1.y - p6.y)**2)**0.5

    return (vertical1 + vertical2) / (2.0 * horizontal)

# ----------------- Main Detection Function -----------------
def start_detection():
    global running, eye_closed_start, alarm_triggered
    running = True
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        eyes_closed = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if ear < 0.78:
                    eyes_closed = True

        if eyes_closed:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            else:
                duration = time.time() - eye_closed_start
                cv2.putText(frame, f"Eyes Closed: {duration:.1f}s", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if duration >= EYE_CLOSED_THRESHOLD and not alarm_triggered:
                    pygame.mixer.music.play(-1)
                    alarm_triggered = True
        else:
            eye_closed_start = None
            if alarm_triggered:
                pygame.mixer.music.stop()
                alarm_triggered = False

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

def stop_detection():
    global running
    running = False

# ----------------- Tkinter GUI -----------------
def start_thread():
    t = threading.Thread(target=start_detection)
    t.start()

root = tk.Tk()
root.title("Driver Drowsiness Detection")
root.geometry("400x200")

start_btn = tk.Button(root, text="Start Detection", font=("Arial", 14), bg="green", fg="white", command=start_thread)
start_btn.pack(pady=20)

stop_btn = tk.Button(root, text="Stop Detection", font=("Arial", 14), bg="red", fg="white", command=stop_detection)
stop_btn.pack(pady=20)

root.mainloop()
