from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time

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

ROTATE_180 = True

# Camera mount area
MOUNT_Y = 440

# ============================================================
# GLOBALS
# ============================================================

lock = threading.Lock()

latest_jpeg = None
background = None

live_candidates = []

selected_candidate = None
calibration_distance = None


# ============================================================
# CAPTURE FRAME
# ============================================================

def capture_frame():

    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    if ROTATE_180:
        frame = cv2.rotate(
            frame,
            cv2.ROTATE_180
        )

    return frame


# ============================================================
# CAPTURE BACKGROUND
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
            frame.astype(np.float32)
        )

        time.sleep(0.05)

    background = np.median(
        np.stack(frames),
        axis=0
    ).astype(np.uint8)

    print("Background captured.")

    return True


# ============================================================
# DETECT OBJECTS
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

    # Difference threshold
    _, mask = cv2.threshold(
        gray,
        30,
        255,
        cv2.THRESH_BINARY
    )

    # Remove tiny noise
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

    # Join parts of same object
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

    # Ignore camera mount
    mask[MOUNT_Y:HEIGHT, :] = 0

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

        # Don't make this too aggressive
        if area < 300:
            continue

        if area > 120000:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 12 or h < 12:
            continue

        if y >= MOUNT_Y:
            continue

        rect_area = w * h

        if rect_area <= 0:
            continue

        fill = area / rect_area

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
        key=lambda x: x["area"],
        reverse=True
    )

    return results[:15]


# ============================================================
# ANALYZE FRESH FRAMES
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

        for d in detections:
            all_detections.append(d)

        time.sleep(0.05)

    if not all_detections:
        print("No candidates found.")
        return []

    # ========================================================
    # GROUP NEARBY DETECTIONS
    # ========================================================

    groups = []

    for d in all_detections:

        matched = False

        for group in groups:

            gx = group[0]["cx"]
            gy = group[0]["cy"]

            distance = (
                (d["cx"] - gx) ** 2 +
                (d["cy"] - gy) ** 2
            ) ** 0.5

            if distance < 50:

                group.append(d)
                matched = True
                break

        if not matched:

            groups.append([d])

    final = []

    for group in groups:

        # Need to appear in multiple frames
        if len(group) < 2:
            continue

        # Average position
        cx = int(
            np.mean(
                [x["cx"] for x in group]
            )
        )

        cy = int(
            np.mean(
                [x["cy"] for x in group]
            )
        )

        w = int(
            np.mean(
                [x["w"] for x in group]
            )
        )

        h = int(
            np.mean(
                [x["h"] for x in group]
            )
        )

        area = int(
            np.mean(
                [x["area"] for x in group]
            )
        )

        fill = float(
            np.mean(
                [x["fill"] for x in group]
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
        key=lambda x: x["area"],
        reverse=True
    )

    return final[:10]


# ============================================================
# DRAW LIVE FEED
# ============================================================

def draw_live(frame, detections):

    output = frame.copy()

    # Mount exclusion
    cv2.line(
        output,
        (0, MOUNT_Y),
        (WIDTH, MOUNT_Y),
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        "MOUNT IGNORED",
        (10, MOUNT_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    for i, d in enumerate(detections):

        x = d["x"]
        y = d["y"]
        w = d["w"]
        h = d["h"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            output,
            (d["cx"], d["cy"]),
            5,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            str(i),
            (x, max(20, y - 5)),
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

                latest_jpeg = encoded.tobytes()
                live_candidates = detections

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
                + frame +
                b"\r\n"
            )

        time.sleep(0.03)


# ============================================================
# WEBSITE
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Atlas Object Calibration</title>

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

</style>

</head>

<body>

<div class="container">

<h1>Atlas Object Calibration</h1>

<img
    id="camera"
    src="/video_feed"
>

<br>

<button
    class="background"
    onclick="captureBackground()">

CAPTURE EMPTY BACKGROUND

</button>

<br>

<input
    id="distance"
    type="number"
    placeholder="Distance mm"
>

<button
    class="analyze"
    onclick="analyzeObject()">

ANALYZE OBJECT

</button>

<div id="status">

Capture the empty background first.

</div>

<div id="results"></div>

</div>


<script>

async function captureBackground() {

    const status =
        document.getElementById("status");

    status.innerHTML =
        "Capturing background...";

    try {

        const response =
            await fetch(
                "/capture_background",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (data.success) {

            status.innerHTML =
                "Background captured. Place the object down.";

        } else {

            status.innerHTML =
                "Background capture failed.";

        }

    } catch (error) {

        status.innerHTML =
            "ERROR: " + error;

    }

}


async function analyzeObject() {

    const distance =
        document.getElementById(
            "distance"
        ).value;

    if (!distance) {

        alert(
            "Enter distance in mm first."
        );

        return;

    }

    const status =
        document.getElementById(
            "status"
        );

    const results =
        document.getElementById(
            "results"
        );

    status.innerHTML =
        "Analyzing 5 fresh camera frames...";

    results.innerHTML = "";

    try {

        const response =
            await fetch(
                "/analyze?distance=" +
                encodeURIComponent(distance)
            );

        const data =
            await response.json();

        if (!response.ok) {

            status.innerHTML =
                "Server error: " +
                JSON.stringify(data);

            return;

        }

        if (
            !data.candidates ||
            data.candidates.length === 0
        ) {

            status.innerHTML =
                "No stable objects detected.";

            return;

        }

        status.innerHTML =
            "Found " +
            data.candidates.length +
            " stable candidates.";

        data.candidates.forEach(
            (c, i) => {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "candidate";

                div.innerHTML = `

                    <h3>
                        Candidate ${i}
                    </h3>

                    Center:
                    (${c.cx}, ${c.cy})

                    <br>

                    Size:
                    ${c.w} × ${c.h}

                    <br>

                    Area:
                    ${c.area}

                    <br>

                    Stable frames:
                    ${c.hits}

                    <br><br>

                    <button
                        class="select"
                        onclick="
                        selectObject(${i})
                        ">

                        SELECT OBJECT

                    </button>

                `;

                results.appendChild(
                    div
                );

            }
        );

    } catch (error) {

        status.innerHTML =
            "REQUEST FAILED: " +
            error;

    }

}


async function selectObject(index) {

    const response =
        await fetch(
            "/select",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    index: index
                })
            }
        );

    const data =
        await response.json();

    if (data.success) {

        document.getElementById(
            "status"
        ).innerHTML =

            "OBJECT SELECTED — " +

            "Pixel center: (" +

            data.candidate.cx +

            ", " +

            data.candidate.cy +

            ")";

    }

}

</script>

</body>

</html>

"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        video_stream(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


@app.route(
    "/capture_background",
    methods=["POST"]
)
def capture_background_route():

    try:

        success = make_background()

        return jsonify({
            "success": success
        })

    except Exception as e:

        print(
            "BACKGROUND ERROR:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/analyze")
def analyze():

    global calibration_distance

    try:

        distance = float(
            request.args.get(
                "distance"
            )
        )

        calibration_distance = distance

        print()
        print("=" * 60)
        print(
            f"ANALYZING OBJECT AT "
            f"{distance} mm"
        )
        print("=" * 60)

        candidates = analyze_fresh_frames()

        print()
        print(
            f"FINAL CANDIDATES: "
            f"{len(candidates)}"
        )

        for i, c in enumerate(candidates):

            print(
                f"[{i}] "
                f"center=({c['cx']},{c['cy']}) "
                f"box={c['w']}x{c['h']} "
                f"area={c['area']} "
                f"hits={c['hits']}"
            )

        print("=" * 60)

        return jsonify({
            "success": True,
            "distance": distance,
            "candidates": candidates
        })

    except Exception as e:

        print(
            "ANALYZE ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route(
    "/select",
    methods=["POST"]
)
def select():

    global selected_candidate

    try:

        data = request.get_json()

        index = int(
            data["index"]
        )

        # Re-run analysis is NOT necessary;
        # use the candidates returned from
        # the latest analysis.

        # For simplicity, get fresh candidates
        candidates = analyze_fresh_frames()

        if (
            index < 0 or
            index >= len(candidates)
        ):

            return jsonify({
                "success": False,
                "error": "Invalid candidate"
            })

        selected_candidate = candidates[index]

        print()
        print("=" * 60)
        print("OBJECT SELECTED")
        print("=" * 60)
        print(
            f"Distance: "
            f"{calibration_distance} mm"
        )
        print(
            f"Center: "
            f"({selected_candidate['cx']}, "
            f"{selected_candidate['cy']})"
        )
        print(
            f"Size: "
            f"{selected_candidate['w']} x "
            f"{selected_candidate['h']}"
        )
        print("=" * 60)

        return jsonify({
            "success": True,
            "candidate":
                selected_candidate
        })

    except Exception as e:

        print(
            "SELECT ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    thread.start()

    print()
    print("=" * 60)
    print("ATLAS OBJECT CALIBRATION")
    print("=" * 60)
    print("Camera: 640x480")
    print("Rotation: 180 degrees")
    print("Port: 5000")
    print()
    print(
        "Open http://<PI-IP>:5000"
    )
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
