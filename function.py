import cv2 as cv


# Function to load the Cascade Classifiers
def cascadeClassifierLoader(cascadeClassifierStringPath):
    return cv.CascadeClassifier(cv.samples.findFile(cascadeClassifierStringPath))


# Image processing : Convert to grayscale & equalize histogram of image
def frameProcessor(frame):
    return cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
