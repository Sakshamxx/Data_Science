import cv2 as cv2

img = cv2.imread("/Users/sakshamchauhan/Downloads/Machine-Learning/Machine_Learning/OpenCV/Lecture-1/dog.jpg")
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Dog Image",img)
cv2.imshow("Gray Dog Image",grayscale)

cv2.waitKey(5000)
cv2.destroyAllWindows