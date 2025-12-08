from ultralytics import YOLO
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load YOLO11 small model
    model = YOLO("yolo11s.pt")

    model.train(
        data="C:/Users/LENOVO/Desktop/Bus_Project/data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=8,
        name="busping_v11",
    )

if __name__ == "__main__":
    main()
