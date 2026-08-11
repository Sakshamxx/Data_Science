import cv2
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# Initialize Camera
# -----------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# -----------------------------
# Face Detection
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    "/Users/sakshamchauhan/Downloads/Machine-Learning/"
    "Machine_Learning/OpenCV/Lecture-1/"
    "haarcascade_frontalface_alt.xml"
)


# -----------------------------
# Load Dataset
# -----------------------------

dataset_path = (
    "/Users/sakshamchauhan/Downloads/Machine-Learning/"
    "Machine_Learning/OpenCV/Lecture-2/data/"
)

face_data = []
labels = []

class_id = 0
names = {}


for fx in os.listdir(dataset_path):

    if fx.endswith(".npy"):

        # Map class ID -> name
        names[class_id] = fx[:-4]

        print("Loaded:", fx)

        # Load face data
        data_item = np.load(
            os.path.join(dataset_path, fx)
        )

        face_data.append(data_item)

        # Create labels
        target = class_id * np.ones(
            (data_item.shape[0],)
        )

        class_id += 1

        labels.append(target)


# -----------------------------
# Combine Dataset
# -----------------------------

face_dataset = np.concatenate(
    face_data,
    axis=0
)

face_labels = np.concatenate(
    labels,
    axis=0
)

print("Dataset:", face_dataset.shape)
print("Labels:", face_labels.shape)
print("Names:", names)


# -----------------------------
# Train KNN
# -----------------------------

clf = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    n_jobs=-1
)

clf.fit(
    face_dataset,
    face_labels
)

print("KNN trained!")


# -----------------------------
# Prediction Stabilization
# -----------------------------

current_prediction = "Detecting..."

candidate_prediction = None
candidate_start_time = None

# New prediction must remain stable
# for 3 seconds before changing
WAIT_TIME = 3.0


# -----------------------------
# Testing
# -----------------------------

current_prediction = "Detecting..."

candidate_prediction = None
candidate_start_time = None

WAIT_TIME = 3.0


# -----------------------------
# Rectangle Stabilization
# -----------------------------

last_face = None

lost_frames = 0
MAX_LOST_FRAMES = 10

# Controls how smoothly rectangle moves
SMOOTHING = 0.4


while True:

    ret, frame = cap.read()

    if not ret:
        continue


    # -----------------------------
    # Resize for faster detection
    # -----------------------------

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.5,
        fy=0.5
    )


    # -----------------------------
    # Grayscale
    # -----------------------------

    gray = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------
    # Face Detection
    # -----------------------------

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )


    # Convert coordinates back
    detected_faces = [
        (x * 2, y * 2, w * 2, h * 2)
        for (x, y, w, h) in detected_faces
    ]


    # -----------------------------
    # Face Found
    # -----------------------------

    if len(detected_faces) > 0:

        # Select largest face
        new_face = sorted(
            detected_faces,
            key=lambda f: f[2] * f[3]
        )[-1]

        x, y, w, h = new_face


        # -----------------------------
        # Smooth Rectangle Movement
        # -----------------------------

        if last_face is None:

            last_face = (
                x,
                y,
                w,
                h
            )

        else:

            old_x, old_y, old_w, old_h = last_face

            x = int(
                old_x * (1 - SMOOTHING)
                + x * SMOOTHING
            )

            y = int(
                old_y * (1 - SMOOTHING)
                + y * SMOOTHING
            )

            w = int(
                old_w * (1 - SMOOTHING)
                + w * SMOOTHING
            )

            h = int(
                old_h * (1 - SMOOTHING)
                + h * SMOOTHING
            )

            last_face = (
                x,
                y,
                w,
                h
            )


        # Face successfully detected
        lost_frames = 0


    # -----------------------------
    # Face Temporarily Lost
    # -----------------------------

    else:

        lost_frames += 1


        # Don't immediately remove rectangle
        if lost_frames > MAX_LOST_FRAMES:

            last_face = None


    # -----------------------------
    # Process Face
    # -----------------------------

    if last_face is not None:

        x, y, w, h = last_face

        offset = 10


        # Safe coordinates
        x1 = max(
            0,
            x - offset
        )

        y1 = max(
            0,
            y - offset
        )

        x2 = min(
            frame.shape[1],
            x + w + offset
        )

        y2 = min(
            frame.shape[0],
            y + h + offset
        )


        # -----------------------------
        # Extract Face
        # -----------------------------

        face_section = frame[
            y1:y2,
            x1:x2
        ]


        # Make sure ROI isn't empty
        if face_section.size != 0:

            face_section = cv2.resize(
                face_section,
                (100, 100)
            )


            # -----------------------------
            # KNN Prediction
            # -----------------------------

            out = clf.predict(
                [face_section.flatten()]
            )

            predicted_name = names[
                int(out[0])
            ]


            # =================================================
            # TEMPORAL STABILIZATION
            # =================================================

            if predicted_name != current_prediction:

                # New candidate
                if candidate_prediction != predicted_name:

                    candidate_prediction = predicted_name

                    candidate_start_time = time.time()


                else:

                    elapsed_time = (
                        time.time()
                        - candidate_start_time
                    )


                    # Change prediction after 3 seconds
                    if elapsed_time >= WAIT_TIME:

                        current_prediction = predicted_name

                        candidate_prediction = None
                        candidate_start_time = None


            else:

                candidate_prediction = None
                candidate_start_time = None


        # =================================================
        # Draw Rectangle
        # =================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )


        # =================================================
        # Display Prediction
        # =================================================

        cv2.putText(
            frame,
            current_prediction,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )


    # -----------------------------
    # Display Camera
    # -----------------------------

    cv2.imshow(
        "Faces",
        frame
    )


    # -----------------------------
    # Quit
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()