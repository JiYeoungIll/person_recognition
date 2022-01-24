import cv2
import numpy as np
import time
import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import threading

capture = cv2.VideoCapture(cv2.CAP_DSHOW)
cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)  #cv2.CAP_DSHOW+1혹은 1로 지정
running = True

def run1():
    global label3, label4

    while running:
        ret1, frame1 = capture.read()  # 카메라로부터 영상을 받아 frame에 저장
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret2, frame2 = cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


        img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        label3.setPixmap(pixmap)

        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        h, w, c = img2.shape
        qImg = QtGui.QImage(img2.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        label4.setPixmap(pixmap)

        if cv2.waitKey(1) == ord('q'):
            return -1
    cap.release()
    capture.release()
    cv2.destroyAllWindows()


def stop():
    global running
    running = False
    print("stoped..")


def start():
    global running
    running = True
    th = threading.Thread(target=run1)
    th.start()
    print("started..")


def onExit():
    print("exit")
    stop()

app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
vbox = QtWidgets.QVBoxLayout()
vbox2 = QtWidgets.QHBoxLayout()
label3 = QtWidgets.QLabel()
label4 = QtWidgets.QLabel()
btn_start = QtWidgets.QPushButton("카메라 켜기")
btn_stop = QtWidgets.QPushButton("카메라 끄기")
vbox2.addWidget(label3)
vbox2.addWidget(label4)
vbox.addLayout(vbox2)
vbox.addWidget(btn_start)
vbox.addWidget(btn_stop)
win.setLayout(vbox)
win.show()

btn_start.clicked.connect(start)
btn_stop.clicked.connect(stop)
app.aboutToQuit.connect(onExit)

sys.exit(app.exec_())