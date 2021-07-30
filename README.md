# AI_virtual-mouse
In this project,I created AI virtual mouse with help of opencv,mediapipe,Autopy

# Use case
In this project using an ai virtual mouse instead of using external mouse.

# Requirements

      pip install opencv-python
      pip install mediapipe
      pip install autopy
      
# How its working

* First,detect and track an hand using mediapipe library.
* Then,Use a hand landmarks to create two classes,one is for moving and another is for clicking on it.
* Using an Autopy library to connect a mouse with hand landmarks.
* Using Autopy you know you sensitivity of mouse to control and adjust it.
      
