from Functions import *

# Loading the frontal face & eyes cascade classifiers obtained from https://github.com/opencv/opencv/tree/master/data
faceCascadeClassifier = cascadeClassifierLoader('data/haarcascades/haarcascade_frontalface_alt.xml')
eyesCascadeClassifier = cascadeClassifierLoader('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

stream = cv.VideoCapture(0)

# Drawing a rectangular bounding box around detected faces and eyes
while stream.isOpened():
    # Read USB camera
    ret, frame = stream.read()

    # Image processing
    processedFrame = frameProcessor(frame)

    # Apply Face Cascade on processed frames, detect faces and return the x-coordinate, y-coordinate, width & height of rectangles
    faceDetection = faceCascadeClassifier.detectMultiScale(processedFrame)

    # Draw bounding boxes around faces
    for (x, y, width, height) in faceDetection:
        startPointBox = (x, y)
        endPointBox = (x + width, y + height)
        colorBox = (255, 0, 0)
        thicknessBox = 2
        frame = cv.rectangle(frame, startPointBox, endPointBox, colorBox, thicknessBox)

        # Apply Eye Cascade on processed frames, detect eyes and return the x-coordinate, y-coordinate, width & height of rectangles
        eyesDetection = eyesCascadeClassifier.detectMultiScale(processedFrame)

        # Draw bounding boxes around eyes
        for (x2, y2, width2, height2) in eyesDetection:
            startPointBox2 = (x2, y2)
            endPointBox2 = (x2 + width2, y2 + height2)
            colorBox = (0, 255, 0)
            thicknessBox = 1
            frame = cv.rectangle(frame, startPointBox2, endPointBox2, colorBox, thicknessBox)

    # Stream display
    cv.imshow('Stream - Face & Eye Detection using Haar Cascade', frame)

    # Break operation
    if cv.waitKey(10) == 27:  # Esc key
        break
