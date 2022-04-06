import cv2 as cv

# Loading the frontal face cascade from local obtained from https://github.com/opencv/opencv/tree/master/data
faceCascade = cv.CascadeClassifier(cv.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml'))

stream = cv.VideoCapture(0)

# Drawing a rectangular bounding box around detected faces
while stream.isOpened():
    # Read USB camera
    ret, frame = stream.read()

    # Image processing : Convert to grayscale & equalize histogram of image
    frameGrayscaleHist = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    # Apply Face Cascade on processed frames, detect faces and return the x-coordinate, y-coordinate, width & height of rectangles
    faceDetection = faceCascade.detectMultiScale(frameGrayscaleHist)

    # Draw bounding boxes
    for (x, y, width, height) in faceDetection:
        startPointBox = (x, y)
        endPointBox = (x + width, y + height)
        colorBox = (255, 0, 0)
        thicknessBox = 2
        frame = cv.rectangle(frame, startPointBox, endPointBox, colorBox, thicknessBox)

    # Stream display
    cv.imshow('Stream - Face Detection using Haar Cascade', frame)

    if cv.waitKey(10) == 27:  # Esc key to break operation
        break
