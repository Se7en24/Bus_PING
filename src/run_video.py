import cv2
from ultralytics import YOLO

def main():
    print("🚀 Loading model...")
    model = YOLO("../runs/detect/busping_v11/weights/best.pt")

    video_path = "C:/Users/LENOVO/Videos/ksrtctest3.mp4" 
    print(f"🎥 Opening video: {video_path}")
    


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video!")
        return

    print("✅ Video opened successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📌 End of video")
            break

        # lower confidence to catch all detections
        results = model(frame, conf=0.2)

        annotated_frame = results[0].plot()

        cv2.imshow("Bus Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✔ Finished!")

if __name__ == "__main__":
    main()
