from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import serial

app = Flask(__name__)

# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)

WIDTH = 640
HEIGHT = 480

ROTATE_90 = True


# ============================================================
# R4 SERIAL
# ============================================================

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200

r4 = None

try:
    r4 = serial.Serial(
        SERIAL_PORT,
        SERIAL_BAUD,
        timeout=1
    )

    time.sleep(2)
    r4.reset_input_buffer()
    r4.reset_output_buffer()

    print("R4 connected on", SERIAL_PORT)

except Exception as e:
    print("WARNING: Could not connect to R4")
    print(e)


def send_to_r4(command):
    global r4

    if r4 is None:
        print("R4 NOT CONNECTED:", command)
        return False

    try:
        message = command.strip() + "\n"

        r4.write(message.encode("utf-8"))
        r4.flush()
        
        time.sleep(0.05)

        print("R4 <-", command)

        return True

    except Exception as e:
        print("R4 SERIAL ERROR:", e)
        try:
            r4.close()
            r4.open()
            time.sleep(2)
        except:
            pass

        return False


# ============================================================
# GLOBALS
# ============================================================

lock = threading.Lock()

latest_jpeg = None
live_detections = []


# ============================================================
# CAMERA
# ============================================================

def capture_frame():
    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    if ROTATE_90:
        frame = cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )

    return frame


# ============================================================
# CAMERA -> ROBOT COORDINATES
# ============================================================

def get_center_camera_x(camera_y):
    points = [
        (103.0, 339.0),
        (183.0, 328.0),
        (263.0, 357.0)
    ]

    if camera_y <= points[0][0]:
        y1, x1 = points[0]
        y2, x2 = points[1]

    elif camera_y >= points[-1][0]:
        y1, x1 = points[-2]
        y2, x2 = points[-1]

    else:
        for i in range(
            len(points) - 1
        ):
            y1, x1 = points[i]
            y2, x2 = points[i + 1]

            if y1 <= camera_y <= y2:
                break

    if y2 == y1:
        return x1

    ratio = (
        (camera_y - y1) /
        (y2 - y1)
    )

    return x1 + ratio * (x2 - x1)


def estimate_robot_y(camera_y):
    camera_points = np.array(
        [
            100.0,  # Top of your workspace view
            400.0   # Bottom of your workspace view
        ],
        dtype=float
    )

    robot_points = np.array(
        [
            200.0,  # Max Y distance
            120.0   # Min Y distance
        ],
        dtype=float
    )

    robot_y = float(
        np.interp(
            camera_y,
            camera_points,
            robot_points,
            left=200.0,
            right=120.0
        )
    )

    return max(120.0, min(200.0, robot_y))


def estimate_robot_x(
    camera_x,
    camera_y
):
    center_x = get_center_camera_x(
        camera_y
    )

    pixel_offset = (
        center_x - camera_x
    )

    pixels_per_mm = 3.5

    x_offset_mm = pixel_offset / pixels_per_mm

    robot_y = estimate_robot_y(
        camera_y
    )

    depth_scale = (
        120.0 /
        max(robot_y, 1.0)
    )

    return float(
        x_offset_mm *
        depth_scale
    )


def camera_to_robot(
    camera_x,
    camera_y
):
    robot_x = estimate_robot_x(
        camera_x,
        camera_y
    )

    robot_y = estimate_robot_y(
        camera_y
    )

    return robot_x, robot_y


# ============================================================
# MANUAL REFERENCE CAPTURE BACKGROUND SUBTRACTION
# ============================================================

stored_background = None

def set_background_reference(frame):
    global stored_background
    stored_background = cv2.GaussianBlur(frame.astype(np.float32), (11, 11), 0)
    print("New empty background reference captured.")


def detect_objects(frame):
    global stored_background

    if stored_background is None:
        stored_background = cv2.GaussianBlur(frame.astype(np.float32), (11, 11), 0)

    current_blur = cv2.GaussianBlur(frame.astype(np.float32), (11, 11), 0)
    
    diff = cv2.absdiff(current_blur, stored_background)
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(diff_gray, 45, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 300 or area > 60000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 15 or h < 15:
            continue

        cx = x + w // 2
        contact_y = y + h

        results.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "cx": int(cx),
            "cy": int(contact_y)
        })

    results.sort(key=lambda item: item["area"], reverse=True)

    return results[:5]


# ============================================================
# DRAW LIVE FEED
# ============================================================

def draw_live(frame, detections):
    output = frame.copy()

    for i, detection in enumerate(detections):
        x = detection["x"]
        y = detection["y"]
        w = detection["w"]
        h = detection["h"]

        color = (0, 255, 0) if i == 0 else (0, 0, 255)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.circle(
            output,
            (detection["cx"], detection["cy"]),
            5,
            color,
            -1
        )

        label = "Target" if i == 0 else f"Obj {i}"
        cv2.putText(
            output,
            label,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():
    global latest_jpeg
    global live_detections

    while True:
        frame = capture_frame()
        detections = detect_objects(frame)
        output = draw_live(frame, detections)

        success, encoded = cv2.imencode(
            ".jpg",
            output,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if success:
            with lock:
                latest_jpeg = encoded.tobytes()
                live_detections = detections

        time.sleep(0.03)


# ============================================================
# VIDEO STREAM
# ============================================================

def video_stream():
    last_frame = None
    while True:
        with lock:
            frame = latest_jpeg

        if frame is not None and frame != last_frame:
            last_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.01)


# ============================================================
# WEBSITE
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Atlas Arm Control</title>
<style>
body {
    background: #111;
    color: white;
    font-family: Arial;
    text-align: center;
}
.container {
    width: 900px;
    max-width: 95%;
    margin: auto;
}
#camera {
    width: 640px;
    max-width: 100%;
    border: 3px solid #444;
}
button {
    padding: 12px 20px;
    margin: 8px;
    font-size: 17px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
.bg-btn {
    background: #0d6efd;
    color: white;
    font-weight: bold;
}
.select {
    background: #198754;
    color: white;
    font-weight: bold;
}
.release {
    background: #ffc107;
    color: black;
    font-weight: bold;
}
#status {
    margin: 20px;
    font-size: 18px;
}
.coordinate {
    background: #183d25;
    margin: 15px;
    padding: 20px;
    border-radius: 8px;
    font-size: 20px;
}
.warning {
    background: #223322;
    padding: 15px;
    margin: 15px;
    border-radius: 8px;
}
</style>
</head>
<body>
<div class="container">
<h1>Atlas Arm Smart Tracking</h1>
<div class="warning">
<b>BACKGROUND REFERENCE LOCK</b>
<br>
Clear the workspace, then click <b>Capture Empty Background</b>. After that, place your object down.
</div>
<img id="camera" src="/video_feed">
<br>
<button class="bg-btn" onclick="captureBackground()">1. CAPTURE EMPTY BACKGROUND</button>
<br>
<button class="select" onclick="moveArmToFirst()">2. GRAB TARGET OBJECT</button>
<button class="release" onclick="releaseArm()">RELEASE & RESET ARM</button>
<div id="status">System ready. Capture empty background first.</div>
<div id="coordinates"></div>
</div>
<script>
async function captureBackground() {
    const status = document.getElementById("status");
    status.innerHTML = "Saving empty background reference...";
    try {
        const response = await fetch("/capture_bg", { method: "POST" });
        const data = await response.json();
        if (data.success) {
            status.innerHTML = "Background reference captured! Place your object down.";
        } else {
            status.innerHTML = "Failed to capture background.";
        }
    } catch (error) {
        status.innerHTML = "REQUEST FAILED: " + error;
    }
}
async function moveArmToFirst() {
    const status = document.getElementById("status");
    status.innerHTML = "Moving arm to target...";
    try {
        const response = await fetch("/move_target", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index: 0 })
        });
        const data = await response.json();
        if (!data.success) {
            status.innerHTML = "Move failed: " + data.error;
            return;
        }
        const target = data.target;
        status.innerHTML = "ARM COMMAND SENT SUCCESSFULLY";
        document.getElementById("coordinates").innerHTML = `
            <div class="coordinate">
                <b>TARGET POSITION</b>
                <br><br>
                Camera: (${target.cx}, ${target.cy})
                <br><br>
                Robot X: <b>${target.robot_x.toFixed(1)} mm</b>
                <br>
                Robot Y: <b>${target.robot_y.toFixed(1)} mm</b>
            </div>
        `;
    } catch (error) {
        status.innerHTML = "REQUEST FAILED: " + error;
    }
}
async function releaseArm() {
    const status = document.getElementById("status");
    status.innerHTML = "Resetting arm position...";
    try {
        const response = await fetch("/release", { method: "POST" });
        const data = await response.json();
        if (data.success) {
            status.innerHTML = "Arm reset command sent.";
        } else {
            status.innerHTML = "Reset failed: " + data.error;
        }
    } catch (error) {
        status.innerHTML = "REQUEST FAILED: " + error;
    }
}
</script>
</body>
</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture_bg", methods=["POST"])
def capture_bg():
    try:
        frame = capture_frame()
        set_background_reference(frame)
        return jsonify({"success": True})
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/move_target", methods=["POST"])
def move_target():
    global live_detections

    try:
        with lock:
            current_detections = list(live_detections)

        if not current_detections:
            return jsonify({
                "success": False,
                "error": "No objects detected against the background. Make sure you captured the empty background first."
            }), 400

        target = current_detections[0]

        robot_x, robot_y = camera_to_robot(target["cx"], target["cy"])
        robot_x -= 40  # calibration offset

        target["robot_x"] = robot_x
        target["robot_y"] = robot_y

        if abs(robot_x) > 150 or robot_y < 50 or robot_y > 400:
            return jsonify({
                "success": False,
                "error": "Detected object is out of safe physical reach bounds."
            }), 400

        command = f"MOVE {robot_x:.1f} {robot_y:.1f} 0.0"
        serial_success = send_to_r4(command)

        return jsonify({
            "success": True,
            "target": target,
            "serial_sent": serial_success
        })

    except Exception as error:
        print("MOVE ERROR:", repr(error))
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/release", methods=["POST"])
def release():
    try:
        command = "RESET"
        serial_success = send_to_r4(command)

        return jsonify({
            "success": True,
            "serial_sent": serial_success
        })

    except Exception as error:
        print("RELEASE ERROR:", repr(error))
        return jsonify({"success": False, "error": str(error)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )
    camera_thread.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
