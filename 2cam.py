import cv2
import threading

capture = cv2.VideoCapture(0)
cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)
running = True

def run1():

    while running:
        ret1, frame1 = capture.read()  # 카메라로부터 영상을 받아 frame에 저장
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.imshow("original1", frame1)
        if cv2.waitKey(1) == ord('q'):
            return -1
    capture.release()
    cv2.destroyAllWindows()
def run2():

    while running:
        ret2, frame2 = cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.imshow("original2", frame2)
        if cv2.waitKey(1) == ord('q'):
                return -1
    cap.release()
    cv2.destroyAllWindows()

th1 = threading.Thread(target=run1)
th2 = threading.Thread(target=run2)
th1.start()
th2.start()
print("started..")


