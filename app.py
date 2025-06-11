from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import asyncio
import json
from ultralytics import YOLO
import torch
import pandas as pd
import os
import seaborn as sns

app = FastAPI()

# 사용할 모델 선택
model_path = 'fire_best.pt'
model = YOLO(model_path)

# 전역 변수로 카메라 객체 관리
camera = None
recording = False
out = None
is_camera_active = False

def get_camera():
    global camera, is_camera_active
    if camera is None and is_camera_active:
        # 웹캠 번호를 0으로 변경 (기본 웹캠)
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("웹캠을 열 수 없습니다. 다른 웹캠 번호를 시도합니다.")
            camera = cv2.VideoCapture(1)
            if not camera.isOpened():
                print("웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인해주세요.")
                return None
        
        print("웹캠이 성공적으로 열렸습니다.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def release_camera():
    global camera, is_camera_active
    if camera is not None:
        camera.release()
        camera = None
        is_camera_active = False
        print("웹캠이 종료되었습니다.")

def generate_frames():
    global recording, out, is_camera_active
    camera = get_camera()
    if camera is None:
        print("카메라를 초기화할 수 없습니다.")
        return
    
    while is_camera_active:
        success, frame = camera.read()
        if not success:
            print("프레임을 읽을 수 없습니다.")
            break
        else:
            try:
                # YOLO로 객체 감지
                results = model(frame)[0]
                
                # 결과 시각화
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if conf > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 녹화 중이면 프레임 저장
                if recording and out is not None:
                    out.write(frame)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"프레임 처리 중 오류 발생: {e}")
                break

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(),
                            media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/start_recording")
async def start_recording():
    global recording, out, is_camera_active
    if not recording:
        is_camera_active = True
        recording = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
        print("녹화가 시작되었습니다.")
    return {"status": "recording started"}

@app.get("/stop_recording")
async def stop_recording():
    global recording, out, is_camera_active
    if recording:
        recording = False
        is_camera_active = False
        if out is not None:
            out.release()
            out = None
            print("녹화가 중지되었습니다.")
        release_camera()
    return {"status": "recording stopped"}

# HTML 템플릿
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>실시간 객체 감지</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            .container {
                max-width: 1280px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            #video-container {
                width: 100%;
                margin-top: 20px;
            }
            #video {
                width: 100%;
                border-radius: 5px;
            }
            .controls {
                margin-top: 20px;
                text-align: center;
            }
            button {
                padding: 10px 20px;
                margin: 0 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>실시간 객체 감지</h1>
            <div id="video-container">
                <img id="video" src="/video_feed" alt="Video feed">
            </div>
            <div class="controls">
                <button id="startBtn" onclick="startRecording()">녹화 시작</button>
                <button id="stopBtn" onclick="stopRecording()" disabled>녹화 중지</button>
            </div>
        </div>
        <script>
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const video = document.getElementById('video');
            
            function startRecording() {
                fetch('/start_recording')
                    .then(response => response.json())
                    .then(data => {
                        console.log('녹화 시작:', data);
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        video.src = '/video_feed?' + new Date().getTime();
                    });
            }
            
            function stopRecording() {
                fetch('/stop_recording')
                    .then(response => response.json())
                    .then(data => {
                        console.log('녹화 중지:', data);
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        video.src = '';
                    });
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 웹캠 캡처 객체 생성 (1은 외부 웹캠)
    cap = cv2.VideoCapture(1)
    
    # 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    recording = False
    out = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break
            
            # YOLO로 객체 감지
            results = model(frame)[0]
            
            # 결과 시각화
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                if conf > 0.5:  # 신뢰도가 0.5 이상인 경우만 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 녹화 중이면 프레임 저장
            if recording and out is not None:
                out.write(frame)
            
            # 프레임을 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # 웹소켓으로 프레임 전송
            await websocket.send_bytes(frame_bytes)
            
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        if out is not None:
            out.release() 