

Meet ATLAS, an autonomous robot built to find, pickup, and sort objects with absolutely no human control. Combining custom manufacturing, programming, and artificial intelligence, ATLAS is designed as a compact bot made to make your life easier.

<img width="1920" height="1080" alt="Screenshot 2026-08-12 at 11 04 58 AM" src="https://github.com/user-attachments/assets/70fe61cd-f5b2-4f25-8cda-0d21e6f71834" />

<a href="[https://example.com](https://cad.onshape.com/documents/8344d425af85321919fd70ba/w/e938afa8e847260ac57abb2e/e/849c61a34d9bbbbca970fb94?resourceType=resourceuserowner&nodeId=683b41990f81c628eee007ce&renderMode=0&uiState=6a8875bcc3a021a91c607942)">
  <img src="https://img.shields.io/badge/ATLAS-111111?style=for-the-badge" alt="ATLAS">
</a>


# KEY FEATURES 
Moves around on its own.
Detects objects in its path using a camera.
Determines which objects it may be able to pick up.
Gives detected objects a confidence score.
Measures how far objects are from the robot.
Uses an arm and claw to pick up objects.
Displays information about what the robot is doing.

# Python V3.11 Required to run firmware
To run code, open a nano inside the Raspberry Pi Python Terminal, and upload the code into the nano, then run the code using python3 (your nano name here) and wait for it to initialize.

# How it works
Atlas is designed around a split between the Raspberry Pi and Arduino: the Pi handles the camera and object detection, while the Arduino handles the robot’s motors, sensors, and servos. This keeps the time-sensitive hardware control separate from the more computationally intensive vision processing and lets each part focus on what it does best.

A major design choice was using a lightweight object-detection model rather than a larger, more accurate model. The Pi 3B+ has limited processing power, so a smaller model makes detection fast enough to be useful while the robot is moving. Atlas also uses confidence scores instead of treating every detection as certain, allowing it to ignore objects it isn't confident it can identify or pick up.

# Credits
Raspberry Pi Foundation — Raspberry Pi hardware and camera ecosystem.
Arduino — UNO R4 WiFi microcontroller platform.
TensorFlow — MobileNet SSD / COCO object-detection model.
COCO Dataset — Object-detection classes and training data.
KiCad — Schematic and electronics design.
OpenCV — Camera and computer-vision processing.
Onshore - Robot hardware design.
