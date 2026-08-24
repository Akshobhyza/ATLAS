import cv2
import serial
import time
import threading

from flask import Flask, Response, redirect
from picamera2 import Picamera2
from gpiozero import LED


# ============================================================
# CONFIG
# ============================================================

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# CAMERA / WEB
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# OBJECT DETECTION
DETECTION_FPS = 30
TARGET_CLASS = "bottle"
CONFIDENCE_THRESHOLD = 0.15

# CENTERING
CENTER_DEADBAND = 45

# BASE LIMITS
BASE_MIN = 30
BASE_CENTER = 90
BASE_MAX = 155

ANGLE_GAIN = 0.18
NEAR_CENTER_GAIN = 0.10
ANGLE_SMOOTHING = 0.25

BASE_COMMAND_INTERVAL = 0.25
MIN_ANGLE_CHANGE = 2

CENTER_CONFIRM_TIME = 0.75


# ============================================================
# ARM
# ============================================================

ARM1_DOWN = 45
ARM1_UP = 90

ARM2_DOWN = 45
ARM2_UP = 90

CLAW_OPEN = 0
CLAW_CLOSED = 90


# ============================================================
# LED
# ============================================================

LED_PIN = 17

led = LED(LED_PIN)


def led_on():
    led.on()


def led_off():
    led.off()


led_off()


# ============================================================
# GLOBAL STATE
# ============================================================

running = True

autonomous_mode = True

reset_in_progress = False
pickup_in_progress = False
pickup_started = False

last_sent_angle = 90
smoothed_angle = 90.0
last_command_time = 0

center_start_time = None


# ============================================================
# CAMERA STATE
# ============================================================

latest_raw_frame = None

raw_frame_lock = threading.Lock()


# ============================================================
# STREAM STATE
# ============================================================

latest_stream_frame = None

stream_frame_lock = threading.Lock()


# ============================================================
# DETECTION STATE
# ============================================================

latest_detection_box = None
latest_detection_confidence = 0.0
latest_detection_error = None

detection_lock = threading.Lock()


# ============================================================
# SERIAL
# ============================================================

arduino = None

serial_lock = threading.Lock()


def connect_arduino():

    global arduino

    while running:

        try:

            print("Connecting to Arduino...")

            arduino = serial.Serial(
                SERIAL_PORT,
                BAUD_RATE,
                timeout=0.1
            )

            time.sleep(3)

            arduino.reset_input_buffer()
            arduino.reset_output_buffer()

            print("Arduino connected.")

            return True

        except Exception as e:

            print("Arduino connection failed:")
            print(e)

            time.sleep(2)

    return False


connect_arduino()


# ============================================================
# SEND COMMAND
# ============================================================

def send_command(command):

    global arduino

    with serial_lock:

        try:

            if arduino is None:

                if not connect_arduino():
                    return False

            arduino.write(
                (command + "\n").encode("ascii")
            )

            arduino.flush()

            print("R4 <-", command)

            return True

        except Exception as e:

            print("Serial error:")
            print(e)

            try:
                arduino.close()
            except:
                pass

            arduino = None

            return False


# ============================================================
# RESET STATE
# ============================================================

def reset_pi_state():

    global pickup_started
    global center_start_time
    global smoothed_angle
    global last_sent_angle
    global last_command_time

    pickup_started = False

    center_start_time = None

    smoothed_angle = 90.0

    last_sent_angle = 90

    last_command_time = time.time()

    led_off()


# ============================================================
# RESET ROBOT
# ============================================================

def reset_robot():

    global reset_in_progress
    global pickup_in_progress

    if reset_in_progress:
        return

    reset_in_progress = True
    pickup_in_progress = False

    print("Resetting ATLAS...")

    reset_pi_state()

    send_command("RESET")

    time.sleep(6)

    reset_pi_state()

    reset_in_progress = False

    print("ATLAS reset complete.")


# ============================================================
# BASE CONTROL
# ============================================================

def send_base_angle(angle):

    global last_sent_angle
    global last_command_time

    if reset_in_progress:
        return

    if pickup_in_progress:
        return

    now = time.time()

    if (
        now - last_command_time
        <
        BASE_COMMAND_INTERVAL
    ):
        return

    angle = int(
        max(
            BASE_MIN,
            min(
                BASE_MAX,
                round(angle)
            )
        )
    )

    if (
        abs(
            angle - last_sent_angle
        )
        <
        MIN_ANGLE_CHANGE
    ):
        return

    if send_command(
        f"BASE {angle}"
    ):

        last_sent_angle = angle
        last_command_time = now


# ============================================================
# MANUAL BASE
# ============================================================

def manual_base(direction):

    global last_sent_angle

    if autonomous_mode:
        return

    if reset_in_progress:
        return

    if pickup_in_progress:
        return

    if direction == "left":

        new_angle = (
            last_sent_angle + 5
        )

    else:

        new_angle = (
            last_sent_angle - 5
        )

    new_angle = max(
        BASE_MIN,
        min(
            BASE_MAX,
            new_angle
        )
    )

    if send_command(
        f"BASE {new_angle}"
    ):

        last_sent_angle = new_angle


# ============================================================
# MANUAL ARM
# ============================================================

def manual_arm(direction):

    if autonomous_mode:
        return

    if reset_in_progress:
        return

    if pickup_in_progress:
        return

    if not hasattr(
        manual_arm,
        "position"
    ):

        manual_arm.position = 90

    if direction == "up":

        manual_arm.position += 5

    else:

        manual_arm.position -= 5

    manual_arm.position = max(
        0,
        min(
            120,
            manual_arm.position
        )
    )

    send_command(
        f"ARM1 {manual_arm.position}"
    )


# ============================================================
# PICKUP
# ============================================================

def pickup():

    global pickup_started
    global pickup_in_progress

    if pickup_started:
        return

    if reset_in_progress:
        return

    pickup_started = True
    pickup_in_progress = True

    print("Starting pickup...")

    try:

        led_off()

        send_command(
            f"CLAW {CLAW_OPEN}"
        )

        time.sleep(0.5)

        send_command(
            f"ARM2 {ARM2_DOWN}"
        )

        time.sleep(0.5)

        send_command(
            f"ARM1 {ARM1_DOWN}"
        )

        time.sleep(0.7)

        send_command(
            f"CLAW {CLAW_CLOSED}"
        )

        time.sleep(0.7)

        send_command(
            f"ARM1 {ARM1_UP}"
        )

        time.sleep(0.7)

        send_command(
            f"ARM2 {ARM2_UP}"
        )

        time.sleep(0.7)

        print("Pickup complete.")

    finally:

        pickup_in_progress = False


# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()


camera_config = (
    picam2.create_video_configuration(
        main={
            "size": (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
            "format": "RGB888"
        },
        controls={
            "FrameRate": CAMERA_FPS
        }
    )
)


picam2.configure(
    camera_config
)

picam2.start()

time.sleep(2)


print(
    f"Camera started: "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} "
    f"@ {CAMERA_FPS} FPS"
)


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():

    global latest_raw_frame

    interval = 1.0 / CAMERA_FPS

    while running:

        start = time.time()

        try:

            frame = picam2.capture_array()

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            # Camera is mounted upside down
            frame = cv2.rotate(
                frame,
                cv2.ROTATE_180
            )

            with raw_frame_lock:

                latest_raw_frame = frame

        except Exception as e:

            print(
                "Camera error:",
                e
            )

        elapsed = (
            time.time() - start
        )

        remaining = (
            interval - elapsed
        )

        if remaining > 0:

            time.sleep(
                remaining
            )


# ============================================================
# AI MODEL
# ============================================================

MODEL = (
    "/home/akshobhyakulkarni/"
    "atlas/models/"
    "MobileNetSSD_deploy.caffemodel"
)


CONFIG = (
    "/home/akshobhyakulkarni/"
    "atlas/models/"
    "MobileNetSSD_deploy.prototxt"
)


CLASSES = [

    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"

]


print("Loading AI model...")


net = cv2.dnn.readNetFromCaffe(
    CONFIG,
    MODEL
)


print("AI model ready.")


# ============================================================
# BOTTLE DETECTION
# ============================================================

def detect_bottle(frame):

    height, width = (
        frame.shape[:2]
    )


    # The camera remains 640x480,
    # but MobileNet receives 300x300.

    small_frame = cv2.resize(
        frame,
        (300, 300)
    )


    blob = cv2.dnn.blobFromImage(
        small_frame,
        0.007843,
        (300, 300),
        127.5
    )


    net.setInput(blob)


    detections = net.forward()


    best_box = None
    best_confidence = 0.0


    for i in range(
        detections.shape[2]
    ):

        confidence = float(
            detections[
                0,
                0,
                i,
                2
            ]
        )


        if (
            confidence
            <
            CONFIDENCE_THRESHOLD
        ):

            continue


        class_id = int(
            detections[
                0,
                0,
                i,
                1
            ]
        )


        if (
            class_id < 0
            or
            class_id >= len(CLASSES)
        ):

            continue


        if (
            CLASSES[class_id]
            !=
            TARGET_CLASS
        ):

            continue


        box = (
            detections[
                0,
                0,
                i,
                3:7
            ]
            *
            [
                width,
                height,
                width,
                height
            ]
        )


        x1, y1, x2, y2 = (
            box.astype(int)
        )


        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            width - 1,
            x2
        )

        y2 = min(
            height - 1,
            y2
        )


        box_width = (
            x2 - x1
        )

        box_height = (
            y2 - y1
        )


        if (
            box_width <= 1
            or
            box_height <= 1
        ):

            continue


        if (
            confidence
            >
            best_confidence
        ):

            best_confidence = (
                confidence
            )

            best_box = (
                x1,
                y1,
                box_width,
                box_height
            )


    return (
        best_box,
        best_confidence
    )


# ============================================================
# DETECTION THREAD
# ============================================================

def detection_loop():

    global latest_detection_box
    global latest_detection_confidence
    global latest_detection_error

    global smoothed_angle
    global center_start_time


    interval = (
        1.0 / DETECTION_FPS
    )


    while running:

        start = time.time()


        with raw_frame_lock:

            if (
                latest_raw_frame
                is None
            ):

                frame = None

            else:

                frame = (
                    latest_raw_frame.copy()
                )


        if frame is not None:

            try:

                (
                    box,
                    confidence
                ) = detect_bottle(
                    frame
                )


                if box is not None:

                    led_on()


                    x, y, w, h = box


                    frame_height, frame_width = (
                        frame.shape[:2]
                    )


                    camera_center = (
                        frame_width / 2
                    )


                    center_x = (
                        x + w / 2
                    )


                    pixel_error = (
                        center_x
                        -
                        camera_center
                    )


                    with detection_lock:

                        latest_detection_box = (
                            box
                        )

                        latest_detection_confidence = (
                            confidence
                        )

                        latest_detection_error = (
                            pixel_error
                        )


                    if (
                        autonomous_mode
                        and
                        not pickup_in_progress
                    ):

                        if (
                            abs(pixel_error)
                            <=
                            CENTER_DEADBAND
                        ):

                            target_angle = 90

                        else:

                            if (
                                abs(pixel_error)
                                <
                                60
                            ):

                                gain = (
                                    NEAR_CENTER_GAIN
                                )

                            else:

                                gain = (
                                    ANGLE_GAIN
                                )


                            target_angle = (
                                90
                                -
                                pixel_error
                                *
                                gain
                            )


                        target_angle = max(
                            BASE_MIN,
                            min(
                                BASE_MAX,
                                target_angle
                            )
                        )


                        smoothed_angle = (
                            (
                                smoothed_angle
                                *
                                (
                                    1
                                    -
                                    ANGLE_SMOOTHING
                                )
                            )
                            +
                            (
                                target_angle
                                *
                                ANGLE_SMOOTHING
                            )
                        )


                        send_base_angle(
                            int(
                                round(
                                    smoothed_angle
                                )
                            )
                        )


                        if (
                            abs(pixel_error)
                            <=
                            CENTER_DEADBAND
                        ):

                            if (
                                center_start_time
                                is None
                            ):

                                center_start_time = (
                                    time.time()
                                )


                            if (
                                time.time()
                                -
                                center_start_time
                                >=
                                CENTER_CONFIRM_TIME
                            ):

                                if (
                                    not
                                    pickup_started
                                ):

                                    threading.Thread(
                                        target=pickup,
                                        daemon=True
                                    ).start()

                        else:

                            center_start_time = None


                else:

                    led_off()

                    center_start_time = None


                    with detection_lock:

                        latest_detection_box = None

                        latest_detection_confidence = (
                            0.0
                        )

                        latest_detection_error = None


            except Exception as e:

                print(
                    "Detection error:",
                    e
                )


        elapsed = (
            time.time()
            -
            start
        )


        remaining = (
            interval
            -
            elapsed
        )


        if remaining > 0:

            time.sleep(
                remaining
            )


# ============================================================
# STREAM THREAD
# ============================================================

def stream_loop():

    global latest_stream_frame


    while running:

        with raw_frame_lock:

            if (
                latest_raw_frame
                is None
            ):

                frame = None

            else:

                frame = (
                    latest_raw_frame.copy()
                )


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        with detection_lock:

            box = (
                latest_detection_box
            )

            confidence = (
                latest_detection_confidence
            )

            error = (
                latest_detection_error
            )


        height, width = (
            frame.shape[:2]
        )


        center = (
            width // 2
        )


        # CENTER LINE

        cv2.line(
            frame,
            (center, 0),
            (center, height),
            (255, 255, 255),
            1
        )


        # DEAD BAND

        cv2.line(
            frame,
            (
                center
                -
                CENTER_DEADBAND,
                0
            ),
            (
                center
                -
                CENTER_DEADBAND,
                height
            ),
            (255, 255, 0),
            1
        )


        cv2.line(
            frame,
            (
                center
                +
                CENTER_DEADBAND,
                0
            ),
            (
                center
                +
                CENTER_DEADBAND,
                height
            ),
            (255, 255, 0),
            1
        )


        # BOTTLE BOX

        if box is not None:

            x, y, w, h = box


            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                2
            )


            object_center = (
                x + w // 2,
                y + h // 2
            )


            cv2.circle(
                frame,
                object_center,
                5,
                (0, 0, 255),
                -1
            )


            cv2.putText(
                frame,
                f"BOTTLE {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            if error is not None:

                cv2.putText(
                    frame,
                    f"Error: {error:.0f}px",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


        else:

            cv2.putText(
                frame,
                "NO BOTTLE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # BASE ANGLE

        cv2.putText(
            frame,
            f"BASE: {last_sent_angle}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # MODE

        if autonomous_mode:

            mode = "AUTONOMOUS"

        else:

            mode = "MANUAL"


        cv2.putText(
            frame,
            mode,
            (
                10,
                height - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        with stream_frame_lock:

            latest_stream_frame = frame


        time.sleep(
            0.001
        )


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


def generate_stream():

    while running:

        with stream_frame_lock:

            if (
                latest_stream_frame
                is None
            ):

                frame = None

            else:

                frame = (
                    latest_stream_frame.copy()
                )


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        success, encoded = (
            cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    50
                ]
            )
        )


        if not success:

            continue


        data = encoded.tobytes()


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            +
            str(
                len(data)
            ).encode()
            +
            b"\r\n\r\n"
            +
            data
            +
            b"\r\n"
        )


# ============================================================
# VIDEO
# ============================================================

@app.route(
    "/video_feed"
)
def video_feed():

    return Response(
        generate_stream(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ============================================================
# MODE
# ============================================================

@app.route(
    "/mode/<mode>",
    methods=["POST"]
)
def change_mode(mode):

    global autonomous_mode
    global center_start_time


    if mode == "auto":

        autonomous_mode = True


    elif mode == "manual":

        autonomous_mode = False


    center_start_time = None


    return redirect("/")


# ============================================================
# MANUAL CONTROLS
# ============================================================

@app.route(
    "/manual/left",
    methods=["POST"]
)
def manual_left():

    manual_base(
        "left"
    )

    return redirect("/")


@app.route(
    "/manual/right",
    methods=["POST"]
)
def manual_right():

    manual_base(
        "right"
    )

    return redirect("/")


@app.route(
    "/manual/up",
    methods=["POST"]
)
def manual_up():

    manual_arm(
        "up"
    )

    return redirect("/")


@app.route(
    "/manual/down",
    methods=["POST"]
)
def manual_down():

    manual_arm(
        "down"
    )

    return redirect("/")


@app.route(
    "/manual/pickup",
    methods=["POST"]
)
def manual_pickup():

    global autonomous_mode


    autonomous_mode = False


    threading.Thread(
        target=pickup,
        daemon=True
    ).start()


    return redirect("/")


# ============================================================
# RESET
# ============================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def website_reset():

    global autonomous_mode


    autonomous_mode = False


    threading.Thread(
        target=reset_robot,
        daemon=True
    ).start()


    return redirect("/")


# ============================================================
# WEBSITE
# ============================================================

@app.route("/")
def index():

    if autonomous_mode:

        mode = "AUTONOMOUS"

    else:

        mode = "MANUAL"


    return f"""
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<title>ATLAS</title>

<style>

body {{
    background:#101010;
    color:white;
    font-family:Arial;
    text-align:center;
    margin:0;
    padding:15px;
}}

h1 {{
    margin:5px;
}}

.video {{
    width:100%;
    max-width:900px;
    margin:auto;
}}

img {{
    width:100%;
    display:block;
    border-radius:10px;
}}

.mode {{
    margin:12px;
    font-size:20px;
    font-weight:bold;
}}

button {{
    border:0;
    border-radius:10px;
    color:white;
    background:#333;
    font-size:18px;
    font-weight:bold;
    padding:15px;
    margin:5px;
}}

.auto {{
    background:#2e7d32;
}}

.manual {{
    background:#1565c0;
}}

.pickup {{
    background:#ef6c00;
}}

.reset {{
    background:#d32f2f;
}}

.controls {{
    margin:15px;
}}

</style>

</head>

<body>

<h1>ATLAS</h1>

<div class="mode">
MODE: {mode}
</div>


<form
action="/mode/auto"
method="POST"
>

<button class="auto">
AUTONOMOUS
</button>

</form>


<form
action="/mode/manual"
method="POST"
>

<button class="manual">
MANUAL
</button>

</form>


<div class="video">

<img src="/video_feed">

</div>


<div class="controls">


<form
action="/manual/up"
method="POST"
>

<button>
⬆️
</button>

</form>


<form
action="/manual/left"
method="POST"
>

<button>
⬅️
</button>

</form>


<form
action="/manual/down"
method="POST"
>

<button>
⬇️
</button>

</form>


<form
action="/manual/right"
method="POST"
>

<button>
➡️
</button>

</form>


</div>


<form
action="/manual/pickup"
method="POST"
>

<button class="pickup">
📦 PICKUP
</button>

</form>


<form
action="/reset"
method="POST"
>

<button class="reset">
🔄 RESET
</button>

</form>


</body>

</html>
"""


# ============================================================
# WEB SERVER
# ============================================================

def start_server():

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )


threading.Thread(
    target=start_server,
    daemon=True
).start()


# ============================================================
# START THREADS
# ============================================================

threading.Thread(
    target=camera_loop,
    daemon=True
).start()


time.sleep(1)


threading.Thread(
    target=detection_loop,
    daemon=True
).start()


threading.Thread(
    target=stream_loop,
    daemon=True
).start()


# ============================================================
# STATUS
# ============================================================

print()
print("==============================")
print("ATLAS ACTIVE")
print("==============================")
print(
    f"Camera: "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} "
    f"@ {CAMERA_FPS} FPS"
)
print(
    f"Detection target: "
    f"{DETECTION_FPS} FPS"
)
print(
    f"Deadband: "
    f"{CENTER_DEADBAND}px"
)
print("Bottle detection: ON")
print("Base limits: 30-155")
print("LED: GPIO17")
print("==============================")
print()


# ============================================================
# KEEP ALIVE
# ============================================================

try:

    while True:

        time.sleep(1)


except KeyboardInterrupt:

    print(
        "Stopping ATLAS..."
    )


finally:

    running = False

    led_off()


    try:

        picam2.stop()

    except:

        pass


    try:

        arduino.close()

    except:

        pass


    print(
        "ATLAS stopped."
    )
