# src/preprocess_all.py
import cv2, os, argparse

def extract_frames(video_path, out_dir, step=10, resize=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % step == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            p = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_f{frame_id:06d}.jpg")
            cv2.imwrite(p, frame)
            saved += 1
        frame_id += 1
    cap.release()
    print(f"Saved {saved} frames from {video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", default="data/videos", help="folder with videos")
    parser.add_argument("--out_dir", default="data/images", help="where frames go")
    parser.add_argument("--step", type=int, default=10, help="save every Nth frame")
    parser.add_argument("--resize", nargs=2, type=int, default=None, help="optional resize WIDTH HEIGHT")
    args = parser.parse_args()

    for fname in os.listdir(args.videos_dir):
        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            extract_frames(
                os.path.join(args.videos_dir, fname),
                args.out_dir,
                step=args.step,
                resize=tuple(args.resize) if args.resize else None
            )

if __name__ == "__main__":
    main()
