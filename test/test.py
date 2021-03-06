# 졸업작품
# 지능형 CCTV
import cv2
import numpy as np
import time
import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui

###
from imutils.video import FPS
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import os # 운영체제 기능 모듈

fps = FPS().start()


# 네트워크 불러오기 - cv2.dnn.readNet
# OpenCv로 딥러닝을 실행하기 위해서는 일단 cv2.dnn.readNet 클래스 객체 생성
# 객체생성에는 훈련된 가중치 / 네트워크 구성을 저장하고 있는 파일이 필요
# cv2.dnn.readNet(model, config=None)
# model : 훈련된 가중치를 저장하고 있는 파일
# confing : 구성파일. 알고리즘에 관한 모든 설정
net = cv2.dnn.readNet("yolo/yolov2-tiny.weights", "yolo/yolov2-tiny.cfg")
# 객체 이름 가져오는 부분
classes = []
with open("tolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# 색상(굳이 필요 없어보임) - 사람 인식할때 그려지는 박스 색상 -- 색상 아닌듯 함
# np.random.unifrom은 NumPy에서 제공하는 균등분포 함수이다.
# 최소값, 최대값, 데이터 개수 순서로 Parameter를 입력해준다.
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# 내장웹캠 연결
cap = cv2.VideoCapture(0)


# 외부웹캠 연결
#cap = cv2.VideoCapture(0)
# instantiate a variable 'p' to keep count of persons
# 숫자를 세기위해 p변수 사용 -- 사용 안해도 될듯
# p = 0
# initialize the writer
# 초기화
writer = None
starting_time = time.time()
frame_id = 0
running = False
# ROI 설정을 위한 마우스 상태, 좌표 초기화
mouse_is_pressing = False
start_x, end_x, start_y, end_y = 0, 0, 0, 0
step = 0
temp = 0


# ROI 설정을 위해 두개의 변수 값을 바꿔주는 함수
def swap(v1, v2):
    global temp
    v1 ,v2 = v2, v1
    return (v1, v2)


# Press The Left Button Of Mouse == Start Position Of ROI
# Release The Left Button Of Mouse == End Position Of ROI
# If Moving The Mouse And Draw Rectangle By ROI Region
def Mouse_Callback(event, x, y, flags, param):
    global step, start_x, end_x, start_y, end_y, mouse_is_pressing
    # Press The Left Button
    if event == cv2.EVENT_LBUTTONDOWN:
        step = 1
        mouse_is_pressing = True
        start_x = x
        start_y = y
    # Moving The Mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        # If Pressing The Mouse
        if mouse_is_pressing:
            step = 2
            end_x = x
            end_y = y
    # Release The Left Button
    elif event == cv2.EVENT_LBUTTONUP:
        step = 3
        mouse_is_pressing = False
        end_x = x
        end_y = y
aaaa = False

# counting 수 초기
count = 0

def run():
    global count
    global aaaa
    global running
    (W, H) = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global label2
    global start_x , start_y, end_x, end_y, step
    while running:
        ret, img = cap.read()
        #img = imutils.resize(img, width=1500)
        count = 0
        global frame_id
        frame_id += 1


        #if W is None or H is None:
            #(H, W) = img.shape[:2]
        # 네트워크 입력 블롭 만들기 - cv2.dnn.blob.FromImage
        # 객체 탐지 부분
        # 입력 영상을 블롭객체로 만들어 추론을 진행 ( 블롭이란? 이진 스케일로 연결된 픽셀 그룹 )
        # 간단히 말해서 자잘한 객체는 노이즈로 처리 - 특정 크기 이상의 큰 객체만 검출
        # scalefactor = 딥러닝 학습 진행할 때, 입력 영상을 픽셀값으로 했는지 정규화 이용했는지 맞게 지정
        # size : 학습할 때, 사용한 영상의 크기
        if aaaa==True:
            frm = img.copy()
            frm = frm[start_y: end_y, start_x: end_x]
            print(frm.shape)

            if frame_id % 5 == 1:
             blob = cv2.dnn.blobFromImage(frm , 0.00392, (416, 416), (0, 0, 0), True, crop=False)
             (H, W) = frm.shape[:2]

        else :
            (H, W) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # 네트워크 입력 설정하기
        net.setInput(blob)
        # 네트워크 순방향 실행(추론)
        outs = net.forward(output_layers)
        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []
        # loop over each of the layer outputs
        for out in outs:
            # loop over each of the detections
            for detection in out:
                # extract the class ID and confidence
                # score는 detection 배열에서 5번째 이후 위치에 있는 값
                scores = detection[5:]
                # scores 배열에서 가장 높은 값을 가지는 값이 confidence
                class_id = np.argmax(scores)
                # 그리고 그때의 위치 인덱스가 class_id
                confidence = scores[class_id]
                # confidence(신뢰도) 지정된 0.6 보다 작은 값은 제외 ( 이 값을 잘 조정해야 검출 정확도 달라짐 )
                # 1에 가까울수록 탐지 정확도 높음
                # 0에 가까울수록 정확도는 낮지만, 탐지되는 수가 많아짐
                if confidence > 0.65:
                    # detection은 scale된 좌상단, 우하단 좌표를 반환이 아니고,
                    # detection object의 중심좌표와 너비/높이를 반환
                    # 원본 영상에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # 노이즈 제거하는 부분
        # 같은 물체에 대한 박스가 많은것을 제거
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

        # 사람 검출
        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                label = str(classes[class_ids[i]])
                if label == 'person':

                    # counting 수 증가
                    count += 1
                    #print("person{}".format(frame_id))
                    # 클래스 ID 및 확률
                    text = "{} : {:.2f}%".format(classes[class_ids[i]], confidences[i])
                    # label text 잘림 방지
                    y = y - 15 if y - 15 > 15 else y + 15
                    # text 출력
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



                else:
                    continue
                # draw a bounding box rectangle and label on the frame
                # color : 배열 나옴 [B,G,R]
                color = [int(c) for c in colors[class_ids[i]]]
                # rectangle(검출영역, 시작점, 종료점, 색상,선굵기 : -1일경우 내부선그리기)
                if aaaa == False:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = label
                    # putText(프레임,텍스트,문자열 위치, 폰트,폰트 크기, 색상,굵기)
                    cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else :
                    img = img[start_y: end_y, start_x: end_x]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = label
                    # putText(프레임,텍스트,문자열 위치, 폰트,폰트 크기, 색상,굵기)
                    cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # counting 결과 출력
        counting_text = "People Counting : {}".format(count)
        cv2.putText(img, counting_text, (10, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        #elapsed_time = time.time() - starting_time
        #fps = frame_id / elapsed_time
        #print(str(round(fps, 2)))
        cv2.namedWindow("Color")
        cv2.setMouseCallback("Color", Mouse_Callback)
        # 파이큐티
        if ret:
            # Press The Left Button
            if step == 1:
                cv2.circle(img, (start_x, start_y), 10, (0, 255, 0), -1)
            # Moving The Mouse
            elif step == 2:
                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
            # Release Of The Mouse
            elif step == 3:
                # If Start X Position Is Bigger Than End X
                if start_x > end_x and start_y < end_y:
                    start_x, end_x = end_x, start_x
                elif start_x > end_x and start_y > end_y:
                    start_y, end_y = end_y, start_y
                    start_x, end_x = end_x, start_x
                elif start_x < end_x and start_y > end_y:
                    start_y, end_y = end_y, start_y




                ROI = img[start_y: end_y, start_x: end_x]
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ROI = cv2.Canny(ROI, 150, 50)
                ROI = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
                img[start_y: end_y, start_x: end_x] = ROI
                aaaa = True

            cv2.imshow("Color", img)
            key = cv2.waitKey(1)
            #esc 누를경우
            if key == 27:
                break

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            label2.setPixmap(pixmap)

        else:
            QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
            print("cannot read frame.")
            break
    cv2.destroyWindow()
    writer.release()
    cap.release()
    print("Thread end.")
def stop():
    global running
    running = False
    print("stoped..")
def start():
    global running
    running = True
    th = threading.Thread(target=run)
    th.start()
    print("started..")
def onExit():
    print("exit")
    stop()
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
vbox = QtWidgets.QVBoxLayout()
label2 = QtWidgets.QLabel()
btn_start = QtWidgets.QPushButton("카메라 켜기")
btn_stop = QtWidgets.QPushButton("카메라 끄기")
vbox.addWidget(label2)
vbox.addWidget(btn_start)
vbox.addWidget(btn_stop)
win.setLayout(vbox)
win.show()
btn_start.clicked.connect(start)
btn_stop.clicked.connect(그만)
app.aboutToQuit.connect(onExit)
sys.exit(app.exec_())