
Meet ATLAS, an autonomous robot built to find, pickup, and sort objects with absolutely no human control. Combining custom manufacturing, programming, and artificial intelligence, ATLAS is designed as a compact bot made to make your life easier.

# KEY FEATURES 
## Project Overview

* Build a small autonomous robot capable of driving around and detecting objects.
* Use an **Arduino UNO R4** with an ultrasonic distance sensor to measure how far away an object is.
* Use a **Raspberry Pi** as the main communication and control system between the different components.
* Use an **Arduino UNO R3** to control the robotic arm and claw.
* When the distance sensor detects an object within **10 cm**, the robot automatically activates the arm.
* The arm slowly lowers toward the object to avoid sudden movements and reduce stress on the gears.
* The claw opens before reaching the object, then closes around it once the arm is lowered.
* After grabbing the object, the arm slowly raises it while the claw remains closed.
* Use two **12 V DC motors** for the robot's drivetrain.
* Use a **Cytron MDD10A dual motor driver** to control the two drive motors.
* Power the motors from a **3S 11.1 V LiPo battery** and use a buck converter to provide regulated power for the electronics.
* Focus on making the robot compact, lightweight, reliable, and capable of operating without constant manual control.
* The overall goal is to create a robot that can **detect, grab, and lift objects autonomously**.
