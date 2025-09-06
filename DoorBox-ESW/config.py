import os

# ============== AWS 설정 ==============
# S3 자격증명
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_HERE"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY_HERE"
AWS_REGION = "ap-northeast-2"
AWS_BUCKET_NAME = "your-bucket-name"

# ============== 디바이스 설정 ==============
DEVICE_ID = "doorbox_001"

# ============== CatchCAM 설정 ==============
RTSP_URL = "rtsp://10.0.0.156/live1.sdp"
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 115200

# ============== 모델 경로 설정 ==============
MODEL_DIR = "models"
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotion3.pth")
AGE_MODEL_PATH = os.path.join(MODEL_DIR, "age1.pt")
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, "gender1.pt")

# ============== 영상 설정 ==============
VIDEO_CLIP_DURATION = 5  # 5초
PRE_BUFFER_DURATION = 2  # 감지 전 2초
POST_BUFFER_DURATION = 3  # 감지 후 3초

# ============== 감지 및 캡처 설정 ==============
DETECTION_DELAY = 0         # 첫 감지 후 0초 대기
CAPTURE_INTERVAL = 5        # 5초마다 캡처
DETECTION_TIMEOUT = 15      # 15초 후 감지 세션 종료

# ============== 업로드 설정 ==============
UPLOAD_BATCH_SIZE = 5
UPLOAD_INTERVAL = 30        # 30초

# ============== 로컬 저장 경로 ==============
LOCAL_OUTPUT_DIR = "output"
LOCAL_FRAMES_DIR = "output/frames"
LOCAL_VIDEOS_DIR = "output/videos"
LOCAL_RESULTS_DIR = "output/results"

# ============== 로그 설정 ==============
LOG_DIR = "logs"
LOG_FILE = "logs/doorbox.log"

# 디렉토리 생성
def create_directories():
    """필요한 디렉토리들을 생성"""
    directories = [
        LOCAL_OUTPUT_DIR,
        LOCAL_FRAMES_DIR,
        LOCAL_VIDEOS_DIR,
        LOCAL_RESULTS_DIR,
        LOG_DIR,
        MODEL_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
