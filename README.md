# iCloudEMS

A real-time computer vision pipeline for campus intelligence. It processes a live video stream frame-by-frame to track attendance, detect motion, and monitor room status.

##  How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


2. Run on a webcam:
    ```bash
   python app.py 0

3. Run on a video file
    ```bash
    python app.py temp_video.mp4

4. Adjust the virtual entry/exit line (if camera angle changes)
    ```bash
    python app.py temp_video.mp4 --line 0.6
