"""
iCloudEMS Assesement

Idea (4 simple steps per frame):
  1. YOLOv8 + ByteTrack 
  2. A virtual line     
  3. Simple OpenCV checks->
"""
import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

#  settings (easy to tune)
BLUR_LIMIT = 60          # Laplacian variance below this = blurry
MOTION_LIMIT = 0.002     # fraction of pixels that changed = motion
TALL_RATIO = 1.4         # box taller than 1.4 x median box = standing
WALK_LIMIT = 0.4         # moved > 40% of box height in ~1s = walking
HISTORY = 10             # frames of position history per person
LIGHT_LIMIT = 90         # avg brightness (0-255) above this = lights on
EMPTY_SECONDS = 5        # how long empty before "lights on, empty" flag
BAND = 50                # pixels around the line ignored (stops flicker)


# small helper functions 
def is_blurry(gray):
    """Sharp images have lots of edges -> high Laplacian variance."""
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_LIMIT


def has_motion(gray, prev_gray):
    """Compare this frame with the previous one."""
    if prev_gray is None:
        return False
    diff = cv2.absdiff(gray, prev_gray)
    changed = (diff > 15).mean()          # fraction of changed pixels
    return changed > MOTION_LIMIT


def side_of_line(cy, line_y):
    """Return 'top', 'bottom', or None while inside the dead band."""
    if cy < line_y - BAND:
        return "top"
    if cy > line_y + BAND:
        return "bottom"
    return None


def is_standing(h, median_h, walking):
    """Standing = walking around OR clearly taller than the other people.
    (Box shape alone fails on high-angle cameras, so compare to the group.)"""
    return walking or h > TALL_RATIO * median_h


# ---------------- main program ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="0", help="0 = webcam, or video path")
    ap.add_argument("--line", type=float, default=0.5, help="line position, 0-1 of frame height")
    ap.add_argument("--reverse", action="store_true", help="entry = moving UP instead of DOWN")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    model = YOLO("yolov8n.pt")            
    
    
    last_side = {}        
    history = defaultdict(lambda: deque(maxlen=HISTORY)) 
    entered_ids = set()   # unique people that entered
    entries = exits = 0
    prev_gray = None
    empty_since = None
    message, message_time = "", 0
    start, frame_no, shown_fps = time.time(), 0, 0.0

    while True:
        # stay real-time: if processing is slower than the video, skip frames
        if is_file:
            target = int((time.time() - start) * fps)
            while frame_no < target:
                if not cap.grab():
                    break
                frame_no += 1
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        t0 = time.time()

        H, W = frame.shape[:2]
        line_y = int(H * args.line)
        gray = cv2.cvtColor(cv2.resize(frame, (320, int(320 * H / W))), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1) detect + track people (class 0 = person)
        res = model.track(frame, persist=True, classes=[0], conf=0.25, imgsz=960,
                          tracker="bytetrack.yaml", verbose=False)[0]

        seated = standing = 0
        ids_now = set()
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            median_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
            for (x1, y1, x2, y2), pid in zip(boxes, ids):
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                ids_now.add(pid)

                # is this person really walking? (moved far over the last ~1 second)
                hist = history[pid]
                hist.append((cx, cy))
                walking = len(hist) == HISTORY and \
                    np.hypot(cx - hist[0][0], cy - hist[0][1]) > WALK_LIMIT * h

                # 2) entry / exit: only WALKING people count (seated people just jitter)
                side = side_of_line(cy, line_y)
                if side is not None:
                    old = last_side.get(pid)
                    if old is not None and old != side and walking:
                        went_down = (side == "bottom")
                        if went_down != args.reverse:   # this direction = entry
                            entries += 1
                            entered_ids.add(pid)
                            message = f"Entry detected (ID {pid})"
                        else:
                            exits += 1
                            message = f"Exit detected (ID {pid})"
                        message_time = time.time()
                        print(message)
                    last_side[pid] = side

                sit = not is_standing(h, median_h, walking)
                seated += sit
                standing += (not sit)

                color = (0, 200, 0) if sit else (0, 140, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"#{pid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        present = len(ids_now)

        # 3) simple frame-level checks
        motion = has_motion(gray, prev_gray)
        prev_gray = gray
        blurry = is_blurry(gray)
        occupied = present > 0

        # bonus: lights on but nobody there for a while
        lights_on = gray.mean() > LIGHT_LIMIT
        if occupied:
            empty_since = None
        elif empty_since is None:
            empty_since = time.time()
        wasted = lights_on and empty_since and (time.time() - empty_since > EMPTY_SECONDS)

        # 4) draw the UI
        cv2.line(frame, (0, line_y), (W, line_y), (255, 0, 0), 2)
        lines = [
            f"{'Motion detected' if motion else 'No motion'}",
            f"{present} currently present",
            f"{len(entered_ids)} unique entries  (entries {entries} / exits {exits})",
            f"Seated: {seated}, Standing/Moving: {standing}",
            "Blurry - flag for review" if blurry else "Clear",
            "Occupied" if occupied else "Empty",
            f"FPS: {shown_fps:.1f}",
        ]
        if wasted:
            lines.append("ALERT: lights on, room empty")
        cv2.rectangle(frame, (0, 0), (430, 30 + 28 * len(lines)), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            bad = text.startswith(("Blurry", "ALERT"))
            cv2.putText(frame, text, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255) if bad else (255, 255, 255), 2)
        if time.time() - message_time < 2:       # show last event for 2 seconds
            cv2.putText(frame, message, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 3)

        cv2.imshow("Smart Campus", cv2.resize(frame, (960, int(960 * H / W))))
        shown_fps = 1 / max(time.time() - t0, 1e-3)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()