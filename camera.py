import cv2

cam = cv2.VideoCapture(6)
result, image = cam.read()
if result: 
  
    # showing result, it take frame name and image  
    # output 
    cv2.imshow("GeeksForGeeks", image) 
  
    # saving image in local storage 
    cv2.imwrite("GeeksForGeeks.png", image) 
  
    # If keyboard interrupt occurs, destroy image  
    # window 
    cv2.waitKey(0) 
    cv2.destroyWindow("GeeksForGeeks") 
  
# If captured image is corrupted, moving to else part 
else: 
    print("No image detected. Please! try again")
cam.release()
