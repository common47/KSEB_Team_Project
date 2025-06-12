"""
화재 감지 웹 애플리케이션
- 실시간 화재 감지 (YOLO 모델)
- 화재 감지 시 음성 알림 (TTS)
- 실시간 비디오 스트리밍
- 비디오 녹화 기능
- 웹 인터페이스
- WebSocket 실시간 통신
"""

# 1. 필요한 라이브러리 임포트
from fastapi import FastAPI, WebSocket  # FastAPI 웹 프레임워크와 WebSocket 지원
from fastapi.responses import HTMLResponse, StreamingResponse  # HTML 응답과 스트리밍 응답
import cv2  # OpenCV - 비디오 캡처 및 이미지 처리
import numpy as np  # 수치 연산
import asyncio  # 비동기 프로그래밍
import json  # JSON 데이터 처리
from ultralytics import YOLO  # YOLO 객체 감지 모델
import uvicorn  # ASGI 서버
import torch  # PyTorch (YOLO 백엔드)
import pandas as pd  # 데이터 처리
import os  # 파일 시스템 작업
import seaborn as sns  # 데이터 시각화
import pyttsx3  # TTS(Text-to-Speech) 엔진
import time  # 시간 관리
import threading  # 스레드 처리

# FastAPI 애플리케이션 초기화
app = FastAPI(title="화재 감지 시스템")

# 2. TTS(음성 알림) 설정
tts_engine = pyttsx3.init()  # TTS 엔진 초기화
tts_engine.setProperty('rate', 150)  # 말하기 속도 설정 (기본값: 200)
tts_engine.setProperty('volume', 1.0)  # 볼륨 설정 (0.0 ~ 1.0)

# TTS 상태 관리 변수
is_speaking = False  # 현재 TTS 실행 중인지 여부
last_tts_time = 0  # 마지막 TTS 실행 시간
TTS_INTERVAL = 5  # TTS 실행 간격 (초)

def speak_text(text):
    """
    TTS를 사용하여 텍스트를 음성으로 변환
    - 마지막 TTS 실행 후 5초가 지났고, 현재 말하고 있지 않은 경우에만 실행
    - 별도의 스레드에서 TTS를 실행하여 메인 프로그램이 블록되지 않도록 함
    """
    global is_speaking, last_tts_time
    current_time = time.time()
    
    if current_time - last_tts_time >= TTS_INTERVAL and not is_speaking:
        is_speaking = True
        last_tts_time = current_time
        
        def speak():
            global is_speaking
            tts_engine.say(text)
            tts_engine.runAndWait()
            is_speaking = False
        
        threading.Thread(target=speak).start()

# 3. YOLO 모델 초기화
model_path = 'fire_best.pt'  # 학습된 YOLO 모델 경로
model = YOLO(model_path)  # YOLO 모델 로드

# 4. 카메라 관리 변수
camera = None  # 카메라 객체
recording = False  # 녹화 상태
out = None  # 녹화 파일 객체
is_camera_active = False  # 카메라 활성화 상태

def get_camera():
    """
    카메라 초기화 및 설정
    - 웹캠 0번 시도 후 실패시 1번 시도
    - 해상도 1280x720 설정
    """
    global camera, is_camera_active
    if camera is None and is_camera_active:
        camera = cv2.VideoCapture(0)  # 기본 웹캠 시도
        if not camera.isOpened():
            print("웹캠을 열 수 없습니다. 다른 웹캠 번호를 시도합니다.")
            camera = cv2.VideoCapture(1)  # 외부 웹캠 시도
            if not camera.isOpened():
                print("웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인해주세요.")
                return None
        
        print("웹캠이 성공적으로 열렸습니다.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def release_camera():
    """카메라 리소스 해제"""
    global camera, is_camera_active
    if camera is not None:
        camera.release()
        camera = None
        is_camera_active = False
        print("웹캠이 종료되었습니다.")

def generate_frames():
    """
    비디오 프레임 생성 및 처리
    - 카메라에서 프레임 읽기
    - YOLO로 객체 감지
    - 화재 감지 시 TTS 알림
    - 녹화 중이면 프레임 저장
    """
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
                fire_detected = False
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if conf > 0.5:  # 신뢰도 50% 이상인 경우만 표시
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 화재가 감지된 경우
                        if model.names[cls] == 'fire' and not fire_detected:
                            fire_detected = True
                            speak_text("화재가 감지되었습니다. 주의하세요.")
                
                # 녹화 중이면 프레임 저장
                if recording and out is not None:
                    out.write(frame)
                
                # 프레임을 JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"프레임 처리 중 오류 발생: {e}")
                break

# 5. FastAPI 엔드포인트
@app.get("/video_feed")
async def video_feed():
    """실시간 비디오 스트림 제공"""
    return StreamingResponse(generate_frames(),
                            media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/start_recording")
async def start_recording():
    """녹화 시작"""
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
    """녹화 중지"""
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
    """메인 페이지 제공"""
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket을 통한 실시간 비디오 스트리밍
    - 객체 감지 결과도 함께 전송
    """
    await websocket.accept()
    
    cap = cv2.VideoCapture(1)
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
                if conf > 0.5:
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

# 6. 메인 실행
if __name__ == "__main__":
    """
    uvicorn 서버 실행
    - host="0.0.0.0": 외부에서도 접속 가능
    - port=8000: 기본 포트
    - reload=True: 코드 변경 시 자동 재시작
    """
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
