import cv2
output_file = 'output_video.avi'
cap=cv2.VideoCapture(output_file)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frames (stream end?). Exiting ...")
        break
    cv2.imshow("Video",frame)
    if cv2.waitKey(0) & 0xFF==ord('d'):
        break
cap.release()
cv2.destroyAllWindows()