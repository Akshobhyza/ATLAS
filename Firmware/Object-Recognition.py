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
        message = command + "\n"

        r4.write(
            message.encode("utf-8")
        )

        r4.flush()

        print("R4 <-", command)

        return True

    except Exception as e:
        print("R4 SERIAL ERROR:", e)

        return False


# ============================================================
# GLOBALS
# ============================================================

lock = threading.Lock()

latest_jpeg = None
background = None

live_candidates = []

last_analyzed_candidates = []

selected_candidate = None

calibration_distance = None


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
    # Adjust center points if needed based on your calibration
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
    # Calibrated points: [Far Pixel, Close Pixel] -> [Far Robot Y (mm), Close Robot Y (mm)]
    camera_points = np.array(
        [
            103.0,  # Far (10 inches)
            263.0   # Close (6 inches)
        ],
        dtype=float
    )

    robot_points = np.array(
        [
            254.0,  # 10 inches in mm
            152.4   # 6 inches in mm
        ],
        dtype=float
    )

    base_y = float(
        np.interp(
            camera_y,
            camera_points,
            robot_points
        )
    )

    # Added +30 mm offset as requested
    return base_y + 30.0


def estimate_robot_x(
    camera_x,
    camera_y
):
    center_x = get_center_camera_x(
        camera_y
    )

    pixel_offset = (
        camera_x - center_x
    )

    # Simplified pixel-to-mm scaling offset
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
# BACKGROUND
# ============================================================

def make_background():
    global background

    frames = []

    print("Capturing background...")

    for i in range(10):
        frame = capture_frame()

        frame = cv2.GaussianBlur(
            frame,
            (9, 9),
            0
        )

        frames.append(
            frame.astype(
                np.float32
            )
        )

        time.sleep(0.05)

    background = np.median(
        np.stack(frames),
        axis=0
    ).astype(np.uint8)

    print("Background captured.")

    return True


# ============================================================
# DETECTION
# ============================================================

def detect_objects(frame):
    if background is None:
        return []

    current = cv2.GaussianBlur(
        frame,
        (9, 9),
        0
    )

    bg = cv2.GaussianBlur(
        background,
        (9, 9),
        0
    )

    diff = cv2.absdiff(
        current,
        bg
    )

    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        30,
        255,
        cv2.THRESH_BINARY
    )

    kernel_small = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    kernel_large = np.ones(
        (13, 13),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area < 300:
            continue

        if area > 120000:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 12 or h < 12:
            continue

        rect_area = w * h

        if rect_area <= 0:
            continue

        fill = (
            area /
            rect_area
        )

        if fill < 0.08:
            continue

        aspect = max(
            w / h,
            h / w
        )

        if aspect > 12:
            continue

        cx = x + w // 2
        cy = y + h // 2

        results.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "cx": int(cx),
            "cy": int(cy),
            "fill": float(fill)
        })

    results.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    return results[:15]


# ============================================================
# MULTI-FRAME ANALYSIS
# ============================================================

def analyze_fresh_frames():
    all_detections = []

    print()
    print("Analyzing fresh frames...")

    for i in range(5):
        frame = capture_frame()

        detections = detect_objects(
            frame
        )

        print(
            f"Analysis frame {i}: "
            f"{len(detections)} candidates"
        )

        all_detections.extend(
            detections
        )

        time.sleep(0.05)

    if not all_detections:
        print("No candidates found.")

        return []

    groups = []

    for detection in all_detections:
        matched = False

        for group in groups:
            gx = group[0]["cx"]
            gy = group[0]["cy"]

            distance = (
                (
                    detection["cx"] - gx
                ) ** 2
                +
                (
                    detection["cy"] - gy
                ) ** 2
            ) ** 0.5

            if distance < 50:
                group.append(
                    detection
                )

                matched = True

                break

        if not matched:
            groups.append(
                [detection]
            )

    final = []

    for group in groups:
        if len(group) < 2:
            continue

        cx = int(
            np.mean(
                [
                    x["cx"]
                    for x in group
                ]
            )
        )

        cy = int(
            np.mean(
                [
                    x["cy"]
                    for x in group
                ]
            )
        )

        w = int(
            np.mean(
                [
                    x["w"]
                    for x in group
                ]
            )
        )

        h = int(
            np.mean(
                [
                    x["h"]
                    for x in group
                ]
            )
        )

        area = int(
            np.mean(
                [
                    x["area"]
                    for x in group
                ]
            )
        )

        fill = float(
            np.mean(
                [
                    x["fill"]
                    for x in group
                ]
            )
        )

        final.append({
            "x": cx - w // 2,
            "y": cy - h // 2,
            "w": w,
            "h": h,
            "area": area,
            "cx": cx,
            "cy": cy,
            "fill": fill,
            "hits": len(group)
        })

    final.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    return final[:10]


# ============================================================
# DRAW LIVE FEED
# ============================================================

def draw_live(
    frame,
    detections
):
    output = frame.copy()

    for i, detection in enumerate(
        detections
    ):
        x = detection["x"]
        y = detection["y"]
        w = detection["w"]
        h = detection["h"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            output,
            (
                detection["cx"],
                detection["cy"]
            ),
            5,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            str(i),
            (
                x,
                max(20, y - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():
    global latest_jpeg
    global live_candidates

    while True:
        frame = capture_frame()

        if background is not None:
            detections = detect_objects(
                frame
            )

        else:
            detections = []

        output = draw_live(
            frame,
            detections
        )

        success, encoded = cv2.imencode(
            ".jpg",
            output,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                75
            ]
        )

        if success:
            with lock:
                latest_jpeg = (
                    encoded.tobytes()
                )

                live_candidates = (
                    detections
                )

        time.sleep(0.02)


# ============================================================
# VIDEO STREAM
# ============================================================

def video_stream():
    while True:
        with lock:
            frame = latest_jpeg

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.03)


# ============================================================
# WEBSITE
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Atlas Arm Test</title>
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
.background {
    background: #2878c8;
    color: white;
}
.analyze {
    background: #198754;
    color: white;
}
.select {
    background: #dc3545;
    color: white;
}
input {
    padding: 10px;
    font-size: 17px;
    width: 120px;
}
#status {
    margin: 20px;
    font-size: 18px;
}
.candidate {
    background: #222;
    margin: 10px;
    padding: 15px;
    border-radius: 8px;
}
.coordinate {
    background: #183d25;
    margin: 15px;
    padding: 20px;
    border-radius: 8px;
    font-size: 20px;
}
.warning {
    background: #553300;
    padding: 15px;
    margin: 15px;
    border-radius: 8px;
}
</style>
</head>
<body>
<div class="container">
<h1>Atlas Arm Position Test</h1>
<div class="warning">
<b>POSITIONING TEST ONLY</b>
<br>
The arm will move to the calculated position.
<br>
Rest position: Whole arm pointing straight up.
</div>
<img id="camera" src="/video_feed">
<br>
<button class="background" onclick="captureBackground()">CAPTURE EMPTY BACKGROUND</button>
<br>
<input id="distance" type="number" placeholder="Distance mm">
<button class="analyze" onclick="analyzeObject()">ANALYZE OBJECT</button>
<div id="status">Capture the empty background first.</div>
<div id="results"></div>
<div id="coordinates"></div>
</div>
<script>
async function captureBackground() {
    const status = document.getElementById("status");
    status.innerHTML = "Capturing background...";
    try {
        const response = await fetch("/capture_background", { method: "POST" });
        const data = await response.json();
        if (data.success) {
            status.innerHTML = "Background captured. Place the object down.";
        } else {
            status.innerHTML = "Background capture failed.";
        }
    } catch (error) {
        status.innerHTML = "ERROR: " + error;
    }
}
async function analyzeObject() {
    const distance = document.getElementById("distance").value;
    if (!distance) {
        alert("Enter distance in mm first.");
        return;
    }
    const status = document.getElementById("status");
    const results = document.getElementById("results");
    status.innerHTML = "Analyzing 5 fresh frames...";
    results.innerHTML = "";
    try {
        const response = await fetch("/analyze?distance=" + encodeURIComponent(distance));
        const data = await response.json();
        if (!response.ok) {
            status.innerHTML = "Server error: " + JSON.stringify(data);
            return;
        }
        if (!data.candidates || data.candidates.length === 0) {
            status.innerHTML = "No stable objects detected.";
            return;
        }
        status.innerHTML = "Found " + data.candidates.length + " stable candidates.";
        data.candidates.forEach((candidate, index) => {
            const div = document.createElement("div");
            div.className = "candidate";
            div.innerHTML = `
                <h3>Candidate ${index}</h3>
                Center: (${candidate.cx}, ${candidate.cy})
                <br>
                Size: ${candidate.w} &times; ${candidate.h}
                <br>
                Area: ${candidate.area}
                <br>
                Stable frames: ${candidate.hits}
                <br>
                <button class="select" onclick="selectObject(${index})">MOVE ARM TO OBJECT</button>
            `;
            results.appendChild(div);
        });
    } catch (error) {
        status.innerHTML = "REQUEST FAILED: " + error;
    }
}
async function selectObject(index) {
    const status = document.getElementById("status");
    status.innerHTML = "Calculating position and moving arm...";
    try {
        const response = await fetch("/select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index: index })
        });
        const data = await response.json();
        if (!data.success) {
            status.innerHTML = "Selection failed: " + data.error;
            return;
        }
        const candidate = data.candidate;
        status.innerHTML = "ARM POSITION COMMAND SENT";
        document.getElementById("coordinates").innerHTML = `
            <div class="coordinate">
                <b>OBJECT POSITION</b>
                <br><br>
                Camera: (${candidate.cx}, ${candidate.cy})
                <br><br>
                Robot X: <b>${candidate.robot_x.toFixed(1)} mm</b>
                <br>
                Robot Y: <b>${candidate.robot_y.toFixed(1)} mm</b>
                <br><br>
                R4 command:
                <br>
                <b>MOVE ${candidate.robot_x.toFixed(1)} ${candidate.robot_y.toFixed(1)} 0.0</b>
            </div>
        `;
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


@app.route("/capture_background", methods=["POST"])
def capture_background_route():
    try:
        success = make_background()
        return jsonify({"success": success})
    except Exception as error:
        print("BACKGROUND ERROR:", repr(error))
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/analyze")
def analyze():
    global calibration_distance
    global last_analyzed_candidates

    try:
        distance = float(request.args.get("distance"))
        calibration_distance = distance

        print()
        print("=" * 60)
        print(f"ANALYZING OBJECT AT {distance} mm")
        print("=" * 60)

        candidates = analyze_fresh_frames()
        last_analyzed_candidates = candidates

        print()
        print(f"FINAL CANDIDATES: {len(candidates)}")
        for i, candidate in enumerate(candidates):
            print(
                f"[{i}] "
                f"center=({candidate['cx']},{candidate['cy']}) "
                f"box={candidate['w']}x{candidate['h']} "
                f"area={candidate['area']} "
                f"hits={candidate['hits']}"
            )
        print("=" * 60)

        return jsonify({
            "success": True,
            "distance": distance,
            "candidates": candidates
        })

    except Exception as error:
        print("ANALYZE ERROR:", repr(error))
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/select", methods=["POST"])
def select():
    global selected_candidate

    try:
        data = request.get_json()

        if data is None:
            return jsonify({"success": False, "error": "No JSON data received"}), 400

        index = int(data["index"])

        if index < 0 or index >= len(last_analyzed_candidates):
            return jsonify({
                "success": False,
                "error": "Invalid candidate. Analyze again."
            }), 400

        selected_candidate = dict(
            last_analyzed_candidates[index]
        )

        # ====================================================
        # CALCULATE ROBOT POSITION
        # ====================================================
        robot_x, robot_y = camera_to_robot(
            selected_candidate["cx"],
            selected_candidate["cy"]
        )
        robot_x -= 40

        selected_candidate["robot_x"] = robot_x
        selected_candidate["robot_y"] = robot_y

        # ====================================================
        # SAFETY LIMITS FOR TESTING (Adjusted for +30 offset)
        # ====================================================
        if abs(robot_x) > 150:
            print("SAFETY STOP: X position too large:", robot_x)
            return jsonify({
                "success": False,
                "error": "Object is outside the safe X test range."
            }), 400

        if robot_y < 50:
            print("SAFETY STOP: Y position too close:", robot_y)
            return jsonify({
                "success": False,
                "error": "Object is too close for the arm test."
            }), 400

        if robot_y > 400:
            print("SAFETY STOP: Y position too far:", robot_y)
            return jsonify({
                "success": False,
                "error": "Object is outside the calibrated Y range."
            }), 400

        # ====================================================
        # SEND POSITION TO R4
        # ====================================================
        command = f"MOVE {robot_x:.1f} {robot_y:.1f} 0.0"
        serial_success = send_to_r4(command)

        print()
        print("=" * 60)
        print("OBJECT SELECTED")
        print("=" * 60)
        print(f"Camera center: ({selected_candidate['cx']}, {selected_candidate['cy']})")
        print(f"Robot X: {robot_x:.1f} mm")
        print(f"Robot Y: {robot_y:.1f} mm")
        print("R4 command:", command)
        print("Serial:", "SENT" if serial_success else "NOT CONNECTED")
        print("=" * 60)

        return jsonify({
            "success": True,
            "candidate": selected_candidate,
            "serial_sent": serial_success
        })

    except Exception as error:
        print("SELECT ERROR:", repr(error))
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

    print()
    print("=" * 60)
    print("ATLAS ARM POSITION TEST")
    print("=" * 60)
    print("Camera: 640x480")
    print("Rotation: 90 degrees")
    print("Port: 5000")
    print()
    print("Servo pins:")
    print("Base  = D9")
    print("Arm1  = D10")
    print("Arm2  = D11")
    print("Claw  = D6")
    print()
    print("Rest Position: Whole arm pointing straight up")
    print()
    print("Open http://<PI-IP>:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
