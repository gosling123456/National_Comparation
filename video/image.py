import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    #conn, addr = s.accept()

    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
      

    #image = cv2.flip(image, 1)
    cv2.imshow('object_detector', image)
cap.release()
cv2.destroyAllWindows()