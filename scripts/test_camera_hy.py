import cv2

def test_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera with index {camera_index} cannot be opened.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow(f'Camera {camera_index}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# test_camera(0)  # Test the first camera
test_camera(2)  # Test the second camera
