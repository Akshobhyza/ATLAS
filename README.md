

Meet ATLAS, an autonomous robot built to find, pickup, and sort objects with absolutely no human control. Combining custom manufacturing, programming, and artificial intelligence, ATLAS is designed as a compact bot made to make your life easier.

<img width="1920" height="1080" alt="Screenshot 2026-08-12 at 11 04 58 AM" src="https://github.com/user-attachments/assets/70fe61cd-f5b2-4f25-8cda-0d21e6f71834" />

<p> For demo links you have to refresh after you click the link </p>
<p> DEMO LINK FOR CHASSIS: https://cad.onshape.com/documents/8344d425af85321919fd70ba/w/e938afa8e847260ac57abb2e/e/849c61a34d9bbbbca970fb94 </p>
<p> DEMO LINK FOR ARM: https://cad.onshape.com/documents/d1ed9956d624a724c1956c00/w/4fc057c3486962ca716a2f76/e/527a2c8adbbe65b72ea021fc?renderMode=0&uiState=6a8922cdfe5bf2908b1126e8</p>

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
Atlas is a bot that works through multiple subsystems that all work together to detect, identify and locate objects. The first subsystem is the Raspberry pi.
1. The Pi is connected to a camera located at the base of the robot arm, and when the camera sees an object, it evaluates it using OpenCV to see it the size is small enough to be picked up by the claw.

2. Next it locates the object using pixels on the cameras frames, trained to accuracy using a data table of pixel rages to distances in mm. 
After it identifies the location of the object, it translates it into x and y coordinates, which it then sends to the Arduino in the form of "MOVE X Y Z" (z is always 0.00mm because it has to reach objects on the ground).

3. After that, the Arduino calculates where the arm should move using inverse kinematics and equations (θbase = atan2(X, Y)), which is the distance from the object using the trigonometry, because the Arduino treats the arm segments as x and y components of a right triangle where the line from the base to the object is the hypotenuse.
   
4. It then calculates the distance, and calculates the elbow angle
   (using cos_theta2 = (R^2 - L1^2 - L2^2) / (2 * L1 * L2) and
   theta_2 = acos(max(-1.0, min(1.0, cos_theta2)))).
   
5. It also calculates the distance from the shoulder to the object using the pythagorean theorem, and finds the angle of the shoulder using the equations cos_beta = (R^2 + L1^2 - L2^2) / (2 * R * L1), beta = acos(max(-1.0, min(1.0, cos_beta))), and theta_1 = alpha + beta where theta1 is the intermediate angle of the shoulder joint.

This is how the bot effectively tracks objects and converts the location of said object into physical movement of the robotic arm using physics, inverse kinematics, and a lot of trigonometry.





One important decision I had to make for Atlas was removing the 3rd microcontroller. Before, I was using the Pi and R4, but also an Arduino Uno R3, but I decided to remove the R3 because I realized that it would take up more space, make all the wiring more difficult, and in general make everything more complicated with no need. Previously the R4 was controlling all sensors and screens, and the R3 was controlling the drivetrain and arm (movement related stuff), but now the R4 controls everything. 

# Credits
Raspberry Pi Foundation — Raspberry Pi hardware and camera ecosystem.
Arduino — UNO R4 WiFi microcontroller platform.
TensorFlow — MobileNet SSD / COCO object-detection model.
COCO Dataset — Object-detection classes and training data.
KiCad — Schematic and electronics design.
OpenCV — Camera and computer-vision processing.
Onshore - Robot hardware design.
