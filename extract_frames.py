import cv2
import os


def extract_frames(video_path, num_frames=30, width=None, height=None):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("data", "frames", basename)
    os.makedirs(output_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(total_frames / num_frames * i) for i in range(num_frames)]

    for idx, frame_num in enumerate(frame_indices):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = vidcap.read()
        if success:
            if width and height:
                frame = cv2.resize(frame, (width, height))
            frame_path = os.path.join(output_dir, f"frame_{idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
    vidcap.release()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("vid_directory", type=str, help="Path to the video directory.")
    parser.add_argument("--width", type=int, help="Width to resize frames to.")
    parser.add_argument("--height", type=int, help="Height to resize frames to.")
    parser.add_argument(
        "--num",
        type=int,
        default=30,
        help="Number of frames to extract from the video.",
    )
    args = parser.parse_args()

    if args.vid_directory:
        for vid in os.listdir(args.vid_directory):
            if vid.endswith(".mp4"):
                video_path = os.path.join(args.vid_directory, vid)
                extract_frames(
                    video_path,
                    width=args.width,
                    height=args.height,
                    num_frames=args.num,
                )

    elif not args.video_path:
        print(
            "Usage: python extract_frames.py <video_path> [--width WIDTH --height HEIGHT]"
        )
    else:
        extract_frames(
            args.video_path, width=args.width, height=args.height, num_frames=args.num
        )
