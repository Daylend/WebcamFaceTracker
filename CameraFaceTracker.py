import numpy as np
import requests
import cv2
import time
import json

# Address of the camera
CAM_URL = "http://192.168.1.30/"
# URL for grabbing a frame
SNAPSHOT_URL = "snapshot.cgi"
# URL for controlling the PTZ
CONTROL_URL = "decoder_control.cgi"
# Credentials (overwritten)
CAM_USER = "admin"
CAM_PASSWORD = "admin"
CREDS_FILENAME = "camcreds.json"
# Facial recognition parameters
PROTOTXT = "deploy.prototxt"
# Neural network model
WEIGHTS = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
# Confidence threshold
CONF_THRESH = 0.35
# Non-maximum suppression threshold
NMS_THRESH = 0.4
# Bounding box to keep the face in
TARGET_X = 150
TARGET_Y = 130
# Allow margin for error to avoid correcting too often
TARGET_RADIUS = 10
# How long each move command moves the camera for, and how long to wait between image polls
WAIT_TIME = 500


# This was an ENUM but python...
PTZ_STOP = 1
PTZ_DOWN = 0
PTZ_UP = 2
PTZ_LEFT = 4
PTZ_RIGHT = 6

# Move camera in specified direction
def movecamera(direction, duration):
    # Craft move and stop URLs for web API
    URL = CAM_URL + CONTROL_URL + "?" + "loginuse=" + CAM_USER + "&" + "loginpas=" + CAM_PASSWORD + \
          "&" + "command=" + str(int(direction)) + "&onestep=0"
    STOPURL = CAM_URL + CONTROL_URL + "?" + "loginuse=" + CAM_USER + "&" + "loginpas=" + CAM_PASSWORD + \
          "&" + "command=" + str(int(PTZ_STOP)) + "&onestep=0"
    # Send move command
    requests.get(URL)
    # convert wait time in ms to s then divide
    time.sleep(duration/1000 / 2)
    # Send stop command
    requests.get(STOPURL)


def main():
    # Read credentials from file as dictionary
    creds = []
    with open(CREDS_FILENAME, "r") as camcreds:
        creds = json.load(camcreds)
    global CAM_USER, CAM_PASSWORD
    CAM_USER = creds["camUser"]
    CAM_PASSWORD = creds["camPassword"]

    while True:
        # Pull camera image from web API
        imgReq = requests.get(CAM_URL + SNAPSHOT_URL + "?" + "user=" + CAM_USER + "&" + "pwd=" + CAM_PASSWORD)
        imageBytes = np.asarray(bytearray(imgReq.content), dtype=np.uint8)
        # Decode image to cv2 format
        image = cv2.imdecode(imageBytes, -1)

        net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)
        (h, w) = image.shape[:2]
        # Resize image and create blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        confidences = []

        # Draws boundary for display (optional)
        cv2.circle(image, (TARGET_X, TARGET_Y), TARGET_RADIUS, (0,255,0), 1)

        # shape function appears to list sizes of each dimension in the array
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= CONF_THRESH:
                boxes.append(detections[0, 0, i, 3:7] * np.array([w, h, w, h]))
                confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

        # For loop is for multiple detections. We only need the most important detection.
        #for i in indices:
        if len(indices) >= 1:
            # Grab most important detection
            i = indices[0][0]
            box = boxes[i]
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10
            centerX = int((startX+endX)/2)
            centerY = int((startY+endY)/2)

            # negative = turn LEFT, positive = turn RIGHT
            centerXDiff = centerX - TARGET_X
            # negative = move DOWN, positive = move UP
            centerYDiff = TARGET_Y - centerY


            # print("cx: ", centerX, "cy: ", centerY, "xdiff: ", centerXDiff, "ydiff: ",
            # centerYDiff, "radius: ", TARGET_RADIUS)

            # elif cause I only want to worry about one direction at a time
            # check for turn in each direction
            if centerXDiff > TARGET_RADIUS:
                movecamera(PTZ_RIGHT, WAIT_TIME)
                #print("RIGHT")
            elif centerXDiff < -1*TARGET_RADIUS:
                movecamera(PTZ_LEFT, WAIT_TIME)
                #print("LEFT")
            elif centerYDiff > TARGET_RADIUS:
                movecamera(PTZ_UP, WAIT_TIME)
                #print("UP")
            elif centerYDiff < -1*TARGET_RADIUS:
                movecamera(PTZ_DOWN, WAIT_TIME)
                #print("DOWN")

            # Print confidences and facial recognition boxes to display (optional)
            text = "{:.2f}%".format(confidences[i]*100)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.rectangle(image, (5+centerX, 5+centerY), (-5+centerX, -5+centerY), (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("test", image)
        cv2.waitKey(WAIT_TIME)

if __name__ == "__main__":
    main()
