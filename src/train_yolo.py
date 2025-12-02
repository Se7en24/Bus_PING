from ultralytics import YOLO
import torch


def main():
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # start from small model for faster iteration
    model = YOLO("yolov8n.pt")  # download automatically the pre-trained tiny model

    model.train(
        data="../data/data.yaml",
        epochs=50,       # increase later if needed
        imgsz=640,
        batch=8,         # lower if GPU memory low
        device=device,   # Use GPU if available
        workers=8,       # Increase workers for faster data loading
        name="busping_v1",
    )


if __name__ == "__main__":
    main()
  