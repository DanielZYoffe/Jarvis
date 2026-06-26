import cv2
import mediapipe as mp
import webcam_access


class HolisticHandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_spec = self.drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

    def process(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.draw_landmarks(output, results)
        return output

    def draw_landmarks(self, image, results):
        if results.left_hand_landmarks:
            self.drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.connection_spec,
            )
            self._draw_landmark_labels(image, results.left_hand_landmarks, "L")

        if results.right_hand_landmarks:
            self.drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.connection_spec,
            )
            self._draw_landmark_labels(image, results.right_hand_landmarks, "R")

    def _draw_landmark_labels(self, image, hand_landmarks, prefix):
        if not hand_landmarks:
            return

        height, width, _ = image.shape
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.putText(
                image,
                f"{prefix}{idx}",
                (x, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def close(self):
        if self.holistic is not None:
            self.holistic.close()
            self.holistic = None


def main():
    tracker = HolisticHandTracker()
    webcam = webcam_access.WebcamAccess(0)

    def callback(frame):
        return tracker.process(frame)

    try:
        webcam.run(callback, title="Holistic Hand Tracking")
    except RuntimeError as exc:
        print(exc)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
