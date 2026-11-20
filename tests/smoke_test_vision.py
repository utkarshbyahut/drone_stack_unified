import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

def main():
    model_path = os.path.expanduser(
        '~/Utkarsh-Sandbox/drone_stack_unified/models/16_k_and_drone_studio_realsense_images_model.engine'
    )
    print(f"Loading TensorRT engine: {model_path}")
    model = YOLO(model_path, task='detect')

    # Initialize 2D Norfair tracker for the OpenCV smoke test
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=60.0,
        hit_counter_max=10,
        initialization_delay=1
    )

    # Create a synthetic 640x480 test frame with a moving white square
    width, height = 640, 480
    box_x, box_y = 100, 200
    vel_x = 5

    print("Starting inference loop. Press 'q' to quit or wait 100 frames...")
    for frame_idx in range(100):
        start_time = time.time()
        
        # Build blank frame and move synthetic target
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        box_x += vel_x
        if box_x > width - 100 or box_x < 50:
            vel_x *= -1
        cv2.rectangle(frame, (box_x, box_y), (box_x + 60, box_y + 60), (255, 255, 255), -1)

        # Run TensorRT YOLO inference
        results = model.predict(frame, conf=0.25, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                detections.append(Detection(points=np.array([[cx, cy]]), scores=np.array([float(conf)])))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Update Norfair tracker
        tracked_objects = tracker.update(detections=detections)
        for obj in tracked_objects:
            pos = obj.estimate[0]
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj.id}", (int(pos[0]) + 10, int(pos[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        fps = 1.0 / max(1e-5, (time.time() - start_time))
        cv2.putText(frame, f"Frame: {frame_idx} | FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display window if X11/display is available, otherwise save output frame
        if os.environ.get("DISPLAY"):
            cv2.imshow("TensorRT + Norfair Smoke Test", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            if frame_idx == 10:
                out_path = os.path.expanduser('~/Utkarsh-Sandbox/drone_stack_unified/tests/smoke_test_out.jpg')
                cv2.imwrite(out_path, frame)
                print(f"No DISPLAY detected (headless SSH). Saved sample frame to {out_path}")
                break

    cv2.destroyAllWindows()
    print("Smoke test completed successfully.")

if __name__ == '__main__':
    main()
