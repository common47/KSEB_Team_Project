from ultralytics import YOLO
import cv2
import torch
import numpy as np
import pandas as pd
import os
import seaborn as sns

# 사용할 모델 선택
# model_path = 'yolo11n.pt'
model_path = 'fire_best.pt'

model = YOLO(model_path)

# 웹캠 캡처 객체 생성 (0은 기본 웹캠)
cap = cv2.VideoCapture(1)

# 웹캠 해상도 설정 (선택사항)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break
    
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 결과 표시
    cv2.imshow('Object Detection - Webcam', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows() 