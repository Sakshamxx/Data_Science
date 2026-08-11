import cv2
import numpy as np
import os

count = 0
face_data = []

cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    "/Users/sakshamchauhan/Downloads/Machine-Learning/"
    "Machine_Learning/OpenCV/Lecture-1/"
    "haarcascade_frontalface_alt.xml"
)

name = input("Enter the Name of the person: ")

while True:

    ret, frame = cap.read()

    if ret == False:
        continue

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray,
        1.1,
        3,
        5
    )

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            [255, 0, 0],
            2
        )

    cv2.imshow("Captured Image", frame)

    # If no face is detected, continue
    if len(faces) == 0:
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        continue

    # Select the largest detected face
    face = sorted(
        faces,
        key=lambda f: f[2] * f[3]
    )[-1]

    x, y, w, h = face

    # Add some padding around the face
    offset = 10

    face_img = frame[
        y - offset:y + h + offset,
        x - offset:x + w + offset
    ]

    # Resize face
    face_img = cv2.resize(face_img, (100, 100))

    cv2.imshow("Face Image", face_img)

    count += 1

    # Store every 10th frame
    if count % 10 == 0:
        face_data.append(face_img.flatten())
        print(len(face_data))

    # Press q to quit
    key = cv2.waitKey(1)

    if key == ord("q"):
        break


# Convert list to NumPy array
face_data = np.array(face_data)

print("Face data shape:", face_data.shape)

# Create data folder relative to this Python file
data_path = os.path.join(
    os.path.dirname(__file__),
    "data"
)

# Create folder if it doesn't exist
os.makedirs(data_path, exist_ok=True)

# Save face data
file_path = os.path.join(
    data_path,
    name + ".npy"
)

np.save(file_path, face_data)

print("Face data saved at:", file_path)

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()