# tracking.py
# Handles video tracking and returns pandas DataFrame

from ultralytics import YOLO
import cv2
import pandas as pd


def track_video(video_path: str, frame_limit: int = 200):
    """
    Runs YOLOv8 tracking on input video.
    Returns dataframe with tracking information.
    """

    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    data = []
    frame_id = 0

    results = model.track(
        source=video_path,
        persist=True,
        conf=0.6,
        iou=0.4,
        classes=[0],
        tracker="bytetrack.yaml",
        imgsz=640,
        stream=True,
        max_det=50
    )

    for r in results:
        boxes = r.boxes

        if boxes is not None:
            for box in boxes:
                if box.id is not None:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    player_id = int(box.id[0])

                    x_center = float((x1 + x2) / 2)
                    y_center = float((y1 + y2) / 2)

                    data.append([
                        frame_id,
                        player_id,
                        x_center,
                        y_center,
                        frame_id / fps
                    ])

        frame_id += 1

        if frame_id > frame_limit:
            break

    cap.release()

    df = pd.DataFrame(
        data,
        columns=[
            "frame_id",
            "player_id",
            "x_center",
            "y_center",
            "timestamp"
        ]
    )

    return df
