# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:31:05 2023

@author: umerr
"""

# import cv2

# # Load the cascade classifier
# face_cascade = cv2.CascadeClassifier("H:\\Uni MPhill Practice\\haarcascade_frontalface_default.xml")

# # Start the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw rectangles around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Show the output
#     cv2.imshow("Faces", frame)

#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam
# cap.release()
# cv2.destroyAllWindows()


import cv2

# Load the cascade classifiers
face_cascade = cv2.CascadeClassifier("H:\\Uni MPhill Practice\\haarcascade_frontalface_default.xml")
hand_cascade = cv2.CascadeClassifier("H:\\Uni MPhill Practice\\palm.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Detect hands
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Hand", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the output
    cv2.imshow("Objects", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
