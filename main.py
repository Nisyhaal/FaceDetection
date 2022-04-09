from Functions import *

# Loading the frontal face & eyes cascade classifiers stored in local obtained from https://github.com/opencv/opencv/tree/master/data
faceCascadeClassifier = cascadeClassifierLoader('data/haarcascades/haarcascade_frontalface_alt.xml')
eyesCascadeClassifier = cascadeClassifierLoader('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

cap = cv.VideoCapture(0)

# Getting width & height of frame for video output configuration
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))

# Video output configuration
videoOutput = cv.VideoWriter('Video_Output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frameWidth, frameHeight))

# Drawing a rectangular bounding box around detected faces and eyes
while True:
    # Read USB camera
    ret, frame = cap.read()

    # Image processing
    processedFrame = frameProcessor(frame)

    # Apply Face Cascade on processed frames, detect faces and return the x-coordinate, y-coordinate, width & height of rectangles
    faceDetection = faceCascadeClassifier.detectMultiScale(processedFrame)

    # Draw bounding boxes around faces
    for (x_face, y_face, width_face, height_face) in faceDetection:
        startPointBox_face = (x_face, y_face)
        endPointBox_face = (x_face + width_face, y_face + height_face)
        colorBox_face = (255, 0, 0)
        thicknessBox_face = 2
        frame = cv.rectangle(frame, startPointBox_face, endPointBox_face, colorBox_face, thicknessBox_face)

        # Apply Eye Cascade on processed frames, detect eyes and return the x-coordinate, y-coordinate, width & height of rectangles
        eyesDetection = eyesCascadeClassifier.detectMultiScale(processedFrame)

        # Draw bounding boxes around eyes
        for (x_eyes, y_eyes, width_eyes, height_eyes) in eyesDetection:
            startPointBox_eyes = (x_eyes, y_eyes)
            endPointBox_eyes = (x_eyes + width_eyes, y_eyes + height_eyes)
            colorBox_eyes = (0, 255, 0)  # green
            thicknessBox_eyes = 1
            frame = cv.rectangle(frame, startPointBox_eyes, endPointBox_eyes, colorBox_eyes, thicknessBox_eyes)

    # Stream display
    cv.imshow('Stream - Face & Eye Detection using Haar Cascade', frame)

    # Stream record
    videoOutput.write(frame)

    # Break operation
    if cv.waitKey(10) == 27:  # Esc key
        break
