import cv2

classifier_file = "car_detector.xml"

video = cv2.VideoCapture("dashcam.mp4")

car_tracker = cv2.CascadeClassifier(classifier_file)

while True:
    video_frame_read, frame = video.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_tracker.detectMultiScale(gray_frame)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    cv2.imshow("Car Detector", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break