import cv2
import numpy as np
import time

# 네트워크 불러오기 - cv2.dnn.readNet
# OpenCv로 딥러닝을 실행하기 위해서는 일단 cv2.dnn.readNet 클래스 객체 생성
# 객체생성에는 훈련된 가중치 / 네트워크 구성을 저장하고 있는 파일이 필요
# cv2.dnn.readNet(model, config=None)
# model : 훈련된 가중치를 저장하고 있는 파일
# confing : 구성파일. 알고리즘에 관한 모든 설정
net = cv2.dnn.readNet("yolov2-tiny.weights", "yolov2-tiny.cfg")

# 객체 이름 가져오는 부분
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 색상(굳이 필요 없어보임) - 사람 인식할때 그려지는 박스 색상
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 내장웹캠 연결
cap = cv2.VideoCapture(0)

# 외부웹캠 연결
# cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)           -CAP_DSHOW+()   () : 인덱스 번호

# instantiate a variable 'p' to keep count of persons
p = 0

# initialize the writer
writer = None
(W, H) = (None, None)
starting_time = time.time()
frame_id = 0

while True:
    ret, frame = cap.read()
    frame_id += 1
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # 네트워크 입력 블롭 만들기 - cv2.dnn.blob.FromImage
    # 객체 탐지 부분
    # 입력 영상을 블롭객체로 만들어 추론을 진행 ( 블롭이란? 이진 스케일로 연결된 픽셀 그룹 )
    # 간단히 말해서 자잘한 객체는 노이즈로 처리 - 특정 크기 이상의 큰 객체만 검출
    # scalefactor = 딥러닝 학습 진행할 때, 입력 영상을 픽셀값으로 했는지 정규화 이용했는지 맞게 지정
    # size : 학습할 때, 사용한 영상의 크기
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
                p = p
            else:
                continue
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = label
            cv2.putText(frame, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    print(str(round(fps, 2)))
    cv2.imshow("Frame", frame)

    # q 입력시, 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        writer.release()
        break

cv2.destroyAllWindows()

#git test