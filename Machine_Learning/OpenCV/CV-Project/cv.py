"""
Gesture-Controlled Browser Launcher
------------------------------------
Uses MediaPipe HandLandmarker (video mode) to detect hand gestures from a
webcam feed and opens a website in your default browser when a gesture is
held steadily.

Gestures:
    Fist        -> LeetCode
    Peace       -> GitHub
    Thumbs Up   -> YouTube
    Open Palm   -> ChatGPT

Run:
    python main.py
Quit:
    press 'q' with the OpenCV window focused
"""

import time
import math
import webbrowser

import cv2
import mediapipe as mp

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

MODEL_PATH = ""

# How many consecutive frames a gesture must be held before it's confirmed.
STABLE_FRAMES_REQUIRED = 8

# Minimum seconds between two site launches (even for the same gesture).
COOLDOWN_SECONDS = 3.0

GESTURE_URLS = {
    "Fist": "https://leetcode.com",
    "Peace": "https://github.com",
    "Thumbs Up": "https://youtube.com",
    "Open Palm": "https://chat.openai.com",
}

# ----------------------------------------------------------------------
# MediaPipe setup
# ----------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


# ----------------------------------------------------------------------
# Gesture recognition
# ----------------------------------------------------------------------

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_open(landmarks, tip_idx, pip_idx, mcp_idx):
    """A non-thumb finger is 'open' if the tip is above both its pip and mcp
    joints (smaller y = higher up in image coordinates)."""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    return tip.y < pip.y and tip.y < mcp.y


def _thumb_open(landmarks):
    """Thumb doesn't fold the same way as other fingers, so use distance
    from the pinky-mcp instead: an extended thumb sits noticeably farther
    from the base of the hand than a folded one, regardless of left/right
    hand or webcam mirroring."""
    tip_dist = _dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
    ip_dist = _dist(landmarks[THUMB_IP], landmarks[PINKY_MCP])
    return tip_dist > ip_dist * 1.1


def classify_gesture(landmarks):
    thumb = _thumb_open(landmarks)
    index = _finger_open(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle = _finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP, INDEX_MCP)
    ring = _finger_open(landmarks, RING_TIP, RING_PIP, INDEX_MCP)
    pinky = _finger_open(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)

    fingers = (thumb, index, middle, ring, pinky)

    # Open Palm: all five fingers extended
    if all(fingers):
        return "Open Palm"

    # Fist: everything folded
    if not any(fingers):
        return "Fist"

    # Peace: index + middle open, thumb/ring/pinky closed
    if index and middle and not ring and not pinky:
        return "Peace"

    # Thumbs Up: only thumb open, and it points clearly upward
    if thumb and not index and not middle and not ring and not pinky:
        if landmarks[THUMB_TIP].y < landmarks[WRIST].y - 0.05:
            return "Thumbs Up"

    return None


# ----------------------------------------------------------------------
# Cooldown / stability tracker
# ----------------------------------------------------------------------

class GestureTrigger:
    """Confirms a gesture only after it's been stable for N frames, then
    enforces a cooldown before it (or any other gesture) can fire again."""

    def __init__(self, stable_frames=STABLE_FRAMES_REQUIRED, cooldown=COOLDOWN_SECONDS):
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self._current_gesture = None
        self._streak = 0
        self._last_fire_time = 0.0
        self._last_confirmed = None  # gesture that triggered last, avoids re-firing on hold

    def update(self, gesture):
        """Feed the latest detected gesture (or None). Returns a gesture
        name if it should trigger an action this frame, else None."""
        if gesture == self._current_gesture:
            self._streak += 1
        else:
            self._current_gesture = gesture
            self._streak = 1

        if gesture is None:
            self._last_confirmed = None
            return None

        stable_enough = self._streak >= self.stable_frames
        cooled_down = (time.time() - self._last_fire_time) >= self.cooldown
        already_fired_for_this_hold = gesture == self._last_confirmed

        if stable_enough and cooled_down and not already_fired_for_this_hold:
            self._last_fire_time = time.time()
            self._last_confirmed = gesture
            return gesture

        return None


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def open_site_for_gesture(gesture):
    url = GESTURE_URLS.get(gesture)
    if url:
        print(f"[ACTION] {gesture} detected -> opening {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[ERROR] Could not open browser: {e}")


def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,  # single hand is enough, and avoids conflicting gestures
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    trigger = GestureTrigger()
    timestamp = 0

    try:
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)  # mirror for natural interaction
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp += 33

                try:
                    result = detector.detect_for_video(mp_image, timestamp)
                except Exception as e:
                    print(f"[ERROR] Detection failed: {e}")
                    continue

                gesture = None

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_landmarks(frame, landmarks)
                    gesture = classify_gesture(landmarks)

                fired = trigger.update(gesture)
                if fired:
                    open_site_for_gesture(fired)

                label = gesture if gesture else "..."
                cv2.putText(
                    frame, f"Gesture: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                )
                cv2.putText(
                    frame, "Press 'q' to quit", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                )

                cv2.imshow("Gesture Browser Launcher", frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

    except FileNotFoundError:
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        print("Download hand_landmarker.task and update MODEL_PATH.")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()