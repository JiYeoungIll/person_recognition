# 졸업작품
# 지능형 CCTV

import cv2
import numpy as np
import time
import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore


# 네트워크 불러오기 - cv2.dnn.readNet
# OpenCv로 딥러닝을 실행하기 위해서는 일단 cv2.dnn.readNet 클래스 객체 생성
# 객체생성에는 훈련된 가중치 / 네트워크 구성을 저장하고 있는 파일이 필요
# cv2.dnn.readNet(model, config=None)
# model : 훈련된 가중치를 저장하고 있는 파일
# confing : 구성파일. 알고리즘에 관한 모든 설정

# 카메라 1
net = cv2.dnn.readNet("yolo/yolov2-tiny.weights", "yolo/yolov2-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# 카메라 2
net2 = cv2.dnn.readNet("yolo/yolov2-tiny.weights", "yolo/yolov2-tiny.cfg")
net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 객체 이름 가져오는 부분
classes = []
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

classes2 = []
with open("yolo/coco.names", "r") as f:
    classes2 = [line.strip() for line in f.readlines()]
layer_names2 = net2.getLayerNames()
output_layers2 = [layer_names2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]

# 색상 - 사람 인식할때 그려지는 박스 색상 // 색상 아닌듯 함
# np.random.unifrom은 NumPy에서 제공하는 균등분포 함수이다.
# 최소값, 최대값, 데이터 개수 순서로 Parameter를 입력해준다.
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 내장웹캠 연결
#cap = cv2.VideoCapture(0)
# 외부웹캠 연결
#cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(cv2.CAP_DSHOW+1)  #cv2.CAP_DSHOW+1혹은 1로 지정

# 초기화
writer = None
starting_time = time.time()
frame_id = 0

# ROI 설정을 위한 마우스 상태, 좌표 초기화
mouse_is_pressing = False
start_x, end_x, start_y, end_y = 0, 0, 0, 0
step = 0
temp = 0

# ROI 설정을 위해 두개의 변수 값을 바꿔주는 함수
def swap(v1, v2):
    global temp
    v1 ,v2 = v2,v1

# 마우스 왼쪽버튼 눌릴때 == 관심영역 시작
# 마우스 왼쪽버튼 땔때 == 관심영역 끝
# 마우스를 이동하여 ROI 영역별로 직사각형을 그리는 경우
def Mouse_Callback(event, x, y, flags, param):
    global step, start_x, end_x, start_y, end_y, mouse_is_pressing
    # 마우스 왼쪽버튼 눌릴때
    if event == cv2.EVENT_LBUTTONDOWN:
        step = 1
        mouse_is_pressing = True
        start_x = x
        start_y = y
    # 마우스 움직일 때
    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 누른 경우
        if mouse_is_pressing:
            step = 2
            end_x = x
            end_y = y
    # 마우스 왼쪽버튼 땔때
    elif event == cv2.EVENT_LBUTTONUP:
        step = 3
        mouse_is_pressing = False
        end_x = x
        end_y = y

# 동작부분
def run():
    global running
    global label2, label3

    (W, H) = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while running:
        ret, img = cap.read()
        global frame_id
        frame_id += 1

        if W is None or H is None:
            (H, W) = img.shape[:2]

        # 네트워크 입력 블롭 만들기 - cv2.dnn.blob.FromImage
        # 객체 탐지 부분
        # 입력 영상을 블롭객체로 만들어 추론을 진행 ( 블롭이란? 이진 스케일로 연결된 픽셀 그룹 )
        # 간단히 말해서 자잘한 객체는 노이즈로 처리 - 특정 크기 이상의 큰 객체만 검출
        # scalefactor = 딥러닝 학습 진행할 때, 입력 영상을 픽셀값으로 했는지 정규화 이용했는지 맞게 지정
        # size : 학습할 때, 사용한 영상의 크기  (416, 416)  이 크기가 적정값
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # 네트워크 입력 설정하기
        net.setInput(blob)

        # 네트워크 순방향 실행(추론)
        outs = net.forward(output_layers)

        # 탐지된 경계 상자, 신뢰도 및 클래스 ID의 목록을 각각 초기화합니다.
        boxes = []
        confidences = []
        class_ids = []
        # 각 레이어 출력 반복
        for out in outs:
            # loop over each of the detections
            for detection in out:
                # 클래스 ID 와 신뢰성 추출
                # score는 detection 배열에서 5번째 이후 위치에 있는 값
                scores = detection[5:]
                class_id = np.argmax(scores)
                # scores 배열에서 가장 높은 값을 가지는 값이 confidence
                # 그리고 그때의 위치 인덱스가 class_id
                confidence = scores[class_id]

                # confidence(신뢰도) 지정된 값보다 작은 값은 제외 ( 이 값을 잘 조정해야 검출 정확도 달라짐 )
                # 1에 가까울수록 탐지 정확도 높음
                # 0에 가까울수록 정확도는 낮지만, 탐지되는 수가 많아짐
                if confidence > 0.7:
                    # detection은 scale된 좌상단, 우하단 좌표를 반환이 아니고,
                    # detection object의 중심좌표와 너비/높이를 반환
                    # 원본 영상에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)

                    # 직사각형 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # 경계 상자 좌표 / 신뢰도 / 클래스 ID 목록 업데이트
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

        # 사람 검출
        if len(indexes) > 0:
            for i in indexes.flatten():
                # 경계박스 좌표 추출
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                label = str(classes[class_ids[i]])
                if label == 'person':
                    print("person")
                else:
                    continue

                # 프레임에 직사각형 박스 및 라벨 표시
                # color : 배열 나옴 [B,G,R]
                color = [int(c) for c in colors[class_ids[i]]]
                # rectangle(검출영역, 시작점, 종료점, 색상,선굵기 : -1일경우 내부선그리기)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = label
                # putText(프레임,텍스트,문자열 위치, 폰트,폰트 크기, 색상,굵기)
                cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        #print(str(round(fps, 2)))
        fps2 = "FPS : %0.1f" % fps
        cv2.putText(img, fps2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        
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
                if start_x > end_x:
                    swap(start_x, end_x)
                    swap(start_y, end_y)

                ROI = img[start_y: end_y, start_x: end_x]
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ROI = cv2.Canny(ROI, 150, 50)
                ROI = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
                img[start_y: end_y, start_x: end_x] = ROI




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
    cap2.release()
    print("Thread end.")

def run2():
    global running2
    (W2, H2) = (cap2.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global label2, label3

    while running2:
        ret2, img2 = cap2.read()
        global frame_id
        frame_id += 1

        if W2 is None or H2 is None:
            (H2, W2) = img2.shape[:2]

        blob2 = cv2.dnn.blobFromImage(img2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


        # 네트워크 입력 설정하기
        net2.setInput(blob2)

        # 네트워크 순방향 실행(추론)
        outs2 = net2.forward(output_layers)

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes2 = []
        confidences2 = []
        class_ids2 = []
        # loop over each of the layer outputs
        for out2 in outs2:
            for detection2 in out2:
                scores2 = detection2[5:]
                class_id2 = np.argmax(scores2)
                confidence2 = scores2[class_id2]

                if confidence2 > 0.65:
                    center_x2 = int(detection2[0] * W2)
                    center_y2 = int(detection2[1] * H2)

                    w2 = int(detection2[2] * W2)
                    h2 = int(detection2[3] * H2)
                    x2 = int(center_x2 - w2 / 2)
                    y2 = int(center_y2 - h2 / 2)

                    boxes2.append([x2, y2, w2, h2])
                    confidences2.append(float(confidence2))
                    class_ids2.append(class_id2)

        indexes2 = cv2.dnn.NMSBoxes(boxes2, confidences2, 0.3, 0.2)

        if len(indexes2) > 0:
            # loop over the indexes we are keeping
            for i in indexes2.flatten():
                # extract the bounding box coordinates
                (x2, y2) = (boxes2[i][0], boxes2[i][1])
                (w2, h2) = (boxes2[i][2], boxes2[i][3])
                label7 = str(classes2[class_ids2[i]])

                if label7 == 'person':
                    print("person")
                else:
                    continue

                # draw a bounding box rectangle and label on the frame
                # color : 배열 나옴 [B,G,R]
                color = [int(c) for c in colors[class_ids2[i]]]
                # rectangle(검출영역, 시작점, 종료점, 색상,선굵기 : -1일경우 내부선그리기)
                cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), color, 2)
                text2 = label7
                cv2.putText(img2, text2, (x2, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        #print(str(round(fps, 2)))

        
        cv2.namedWindow("Color")
        cv2.setMouseCallback("Color", Mouse_Callback)
    
        # 파이큐티
        if ret2:
            # Press The Left Button
            if step == 1:
                cv2.circle(img2, (start_x, start_y), 10, (0, 255, 0), -1)

            # Moving The Mouse
            elif step == 2:
                cv2.rectangle(img2, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)

            # Release Of The Mouse
            elif step == 3:
                # If Start X Position Is Bigger Than End X
                if start_x > end_x:
                    swap(start_x, end_x)
                    swap(start_y, end_y)

                ROI = img2[start_y: end_y, start_x: end_x]
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ROI = cv2.Canny(ROI, 150, 50)
                ROI = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
                img2[start_y: end_y, start_x: end_x] = ROI



            cv2.imshow("Color", img2)
            key = cv2.waitKey(1)
            #esc 누를경우
            if key == 27:
                break

            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            h2, w2, c2 = img2.shape
            qImg = QtGui.QImage(img2.data, w2, h2, w2 * c2, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            label3.setPixmap(pixmap)


        else:
            QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
            print("cannot read frame.")
            break
        

    cv2.destroyWindow()
    writer.release()
    cap.release()
    cap2.release()
    print("Thread end.")

def stop():
    global running, running2
    running = False
    running2 = False
    print("stoped..")


def start():
    global running, running2
    running = True
    running2 = True
    th = threading.Thread(target=run)
    th2 = threading.Thread(target=run2)
    th.start()
    th2.start()
    print("started..")


def onExit():
    print("exit")
    stop()

app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
vbox = QtWidgets.QVBoxLayout()
vbox2 = QtWidgets.QHBoxLayout()
label2 = QtWidgets.QLabel()
label3 = QtWidgets.QLabel()
btn_start = QtWidgets.QPushButton("카메라 켜기")
btn_stop = QtWidgets.QPushButton("카메라 끄기")
vbox2.addWidget(label2)
vbox2.addWidget(label3)

vbox.addLayout(vbox2)
vbox.addWidget(btn_start)
vbox.addWidget(btn_stop)
win.setLayout(vbox)
win.show()


btn_start.clicked.connect(start)
btn_stop.clicked.connect(stop)
app.aboutToQuit.connect(onExit)

sys.exit(app.exec_())