import cv2

classifier_car_file = "car_detector.xml"
classifier_pedestrian_file = "haarcascade_fullbody.xml"

video = cv2.VideoCapture("cars-pedestrians.mp4")

car_tracker = cv2.CascadeClassifier(classifier_car_file)
pedestrian_tracker = cv2.CascadeClassifier(classifier_pedestrian_file)

while True:
    video_frame_read, frame = video.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_tracker.detectMultiScale(gray_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(gray_frame)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
    cv2.imshow("Car and Pedestrian Detector", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break