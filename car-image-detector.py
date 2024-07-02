import cv2

img_file = "cars.png"

classifier_file = "car_detector.xml"

car_tracker = cv2.CascadeClassifier(classifier_file)

img = cv2.imread(img_file)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = car_tracker.detectMultiScale(gray_img)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Car Detector", img)
cv2.waitKey()
