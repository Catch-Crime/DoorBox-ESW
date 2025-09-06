import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, mobilenet_v3_small
import timm  # GhostNet을 위해 필요
import serial
import threading
import time
import json
import os
import boto3
from datetime import datetime, timezone
import pytz  # 시간대 설정용
from collections import deque
import logging
from botocore.exceptions import ClientError
import warnings

# Warning 메시지 완전히 숨기기
warnings.filterwarnings("ignore")
import torchvision  

# 설정 파일 import
import config

# ================= PIR, OLED, RGB LED 관련 추가 =================
# GPIO library detection
try:
    from gpiozero import LED, MotionSensor
    GPIO_METHOD = "gpiozero"
    print("✅ Using gpiozero library")
except ImportError:
    GPIO_METHOD = "direct"
    print("✅ Using direct GPIO control")

# I2C OLED library
try:
    import smbus2 as smbus
    I2C_AVAILABLE = True
    print("✅ Using smbus2 for OLED")
except ImportError:
    try:
        import smbus
        I2C_AVAILABLE = True
        print("✅ Using smbus for OLED")
    except ImportError:
        I2C_AVAILABLE = False
        print("❌ I2C library not available")

from PIL import Image, ImageDraw, ImageFont

class RGBController:
    """RGB LED Controller (Common Cathode)"""
    
    def __init__(self):
        self.pins = {'RED': 18, 'GREEN': 23, 'BLUE': 24}  # GPIO numbers
        
        if GPIO_METHOD == "gpiozero":
            self._init_gpiozero()
        else:
            self._init_direct()
        
        # Turn off all LEDs initially
        self.set_rgb(0, 0, 0)
        print("✅ RGB LED initialized (Common Cathode)")
    
    def _init_gpiozero(self):
        """Initialize using gpiozero"""
        self.red_led = LED(18)    # Pin 12
        self.green_led = LED(23)  # Pin 16  
        self.blue_led = LED(24)   # Pin 18
    
    def _init_direct(self):
        """Initialize using direct GPIO control"""
        # Cleanup existing GPIO
        for pin in self.pins.values():
            try:
                with open('/sys/class/gpio/unexport', 'w') as f:
                    f.write(str(pin))
            except:
                pass
        
        # Export and setup
        for pin in self.pins.values():
            with open('/sys/class/gpio/export', 'w') as f:
                f.write(str(pin))
            time.sleep(0.1)
            with open(f'/sys/class/gpio/gpio{pin}/direction', 'w') as f:
                f.write('out')
    
    def _set_pin_value(self, pin, value):
        """Set GPIO pin value (direct method)"""
        try:
            with open(f'/sys/class/gpio/gpio{pin}/value', 'w') as f:
                f.write(str(value))
        except Exception as e:
            print(f"GPIO {pin} error: {e}")
    
    def set_rgb(self, r, g, b):
        """Set RGB color (0-255 each)"""
        # Convert 0-255 to 0/1 (threshold at 128)
        red_val = 1 if r > 128 else 0
        green_val = 1 if g > 128 else 0
        blue_val = 1 if b > 128 else 0
        
        if GPIO_METHOD == "gpiozero":
            # gpiozero: on()=HIGH, off()=LOW
            if red_val:
                self.red_led.on()
            else:
                self.red_led.off()
                
            if green_val:
                self.green_led.on()
            else:
                self.green_led.off()
                
            if blue_val:
                self.blue_led.on()
            else:
                self.blue_led.off()
        else:
            # Direct GPIO control
            self._set_pin_value(self.pins['RED'], red_val)
            self._set_pin_value(self.pins['GREEN'], green_val)
            self._set_pin_value(self.pins['BLUE'], blue_val)
    
    def set_color_by_name(self, color_name):
        """Set color by predefined names"""
        colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "purple": (255, 0, 255),
            "off": (0, 0, 0)
        }
        if color_name in colors:
            self.set_rgb(*colors[color_name])
    
    def blink_purple(self, times=3, interval=0.3):
        """보라색 LED를 지정된 횟수만큼 깜빡임"""
        for i in range(times):
            self.set_rgb(255, 0, 255)  # 보라색 켜기
            time.sleep(interval)
            self.set_rgb(0, 0, 0)      # LED 끄기
            time.sleep(interval)
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        try:
            self.set_rgb(0, 0, 0)
        except:
            pass  # 이미 종료된 경우 무시
        
        if GPIO_METHOD == "gpiozero":
            try:
                if hasattr(self, 'red_led') and self.red_led:
                    self.red_led.close()
                if hasattr(self, 'green_led') and self.green_led:
                    self.green_led.close()
                if hasattr(self, 'blue_led') and self.blue_led:
                    self.blue_led.close()
            except:
                pass
        else:
            for pin in self.pins.values():
                try:
                    with open('/sys/class/gpio/unexport', 'w') as f:
                        f.write(str(pin))
                except:
                    pass

class OLEDDisplay:
    """Simple OLED Display Controller"""
    
    def __init__(self):
        if not I2C_AVAILABLE:
            raise Exception("I2C library not available")
        
        self.bus = smbus.SMBus(1)
        self.addr = 0x3C
        self.width = 128
        self.height = 64
        
        # Initialize OLED
        init_sequence = [
            0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40,
            0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x12,
            0x81, 0x7F, 0xD9, 0x22, 0xDB, 0x20, 0xA4, 0xA6, 0xAF
        ]
        
        for cmd in init_sequence:
            self.bus.write_byte_data(self.addr, 0x00, cmd)
        
        # Load font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
        
        self.clear()
        print("✅ OLED display initialized")
    
    def clear(self):
        """Clear the display"""
        for page in range(8):
            self.bus.write_byte_data(self.addr, 0x00, 0xB0 + page)
            self.bus.write_byte_data(self.addr, 0x00, 0x00)
            self.bus.write_byte_data(self.addr, 0x00, 0x10)
            
            for col in range(128):
                self.bus.write_byte_data(self.addr, 0x40, 0x00)
    
    def display_text(self, lines):
        """Display text lines with improved error handling"""
        if not hasattr(self, 'display_lock'):
            self.display_lock = threading.Lock()
        
        with self.display_lock:
            try:
                image = Image.new("1", (self.width, self.height))
                draw = ImageDraw.Draw(image)
                
                y = 0
                for i, line in enumerate(lines[:6]):
                    font = self.font if i == 0 else self.font_small
                    # 문자열 길이 제한으로 깨짐 방지
                    line_str = str(line)[:20] if len(str(line)) > 20 else str(line)
                    draw.text((0, y), line_str, font=font, fill=1)
                    y += 11 if i == 0 else 10
                
                pixels = list(image.getdata())
                
                # OLED 업데이트를 더 안정적으로
                for page in range(8):
                    try:
                        self.bus.write_byte_data(self.addr, 0x00, 0xB0 + page)
                        self.bus.write_byte_data(self.addr, 0x00, 0x00)
                        self.bus.write_byte_data(self.addr, 0x00, 0x10)
                        
                        for col in range(128):
                            byte_val = 0
                            for bit in range(8):
                                pixel_y = page * 8 + bit
                                if pixel_y < 64:
                                    pixel_idx = pixel_y * 128 + col
                                    if pixel_idx < len(pixels) and pixels[pixel_idx]:
                                        byte_val |= (1 << bit)
                            
                            self.bus.write_byte_data(self.addr, 0x40, byte_val)
                    except Exception as e:
                        # 페이지 업데이트 실패시 계속 진행
                        continue
                        
            except Exception as e:
                # 전체 디스플레이 업데이트 실패시 조용히 무시
                pass

class PIRSensor:
    """PIR Motion Sensor Controller"""
    
    def __init__(self):
        self.pir_pin = 14  # GPIO14 (Pin 8)
        self.last_detection = None
        self.detection_count = 0
        self.motion_callback = None
        
        if GPIO_METHOD == "gpiozero":
            self._init_gpiozero()
        else:
            self._init_direct()
        
        print("✅ PIR sensor initialized (GPIO14)")
    
    def _init_gpiozero(self):
        """Initialize PIR using gpiozero"""
        self.pir_sensor = MotionSensor(self.pir_pin)
        self.pir_sensor.when_motion = self._motion_detected
    
    def _init_direct(self):
        """Initialize PIR using direct GPIO"""
        try:
            with open('/sys/class/gpio/unexport', 'w') as f:
                f.write(str(self.pir_pin))
        except:
            pass
        
        with open('/sys/class/gpio/export', 'w') as f:
            f.write(str(self.pir_pin))
        time.sleep(0.1)
        
        with open(f'/sys/class/gpio/gpio{self.pir_pin}/direction', 'w') as f:
            f.write('in')
        
        self.last_state = 0
    
    def set_callback(self, callback):
        """Set motion detection callback"""
        self.motion_callback = callback
    
    def _motion_detected(self):
        """Motion detection callback"""
        self.detection_count += 1
        self.last_detection = datetime.now()
        
        print(f"🚶 PIR Motion detected! Count: {self.detection_count}")
        print(f"   Time: {self.last_detection.strftime('%H:%M:%S')}")
        
        if self.motion_callback:
            self.motion_callback()
    
    def check_motion_direct(self):
        """Check motion for direct GPIO method"""
        if GPIO_METHOD != "direct":
            return False
            
        try:
            with open(f'/sys/class/gpio/gpio{self.pir_pin}/value', 'r') as f:
                current_state = int(f.read().strip())
            
            if current_state == 1 and self.last_state == 0:
                self.last_state = current_state
                self._motion_detected()
                return True
            
            self.last_state = current_state
            return False
            
        except Exception as e:
            print(f"PIR read error: {e}")
            return False
    
    def get_status(self):
        """Get current PIR status"""
        if GPIO_METHOD == "gpiozero":
            try:
                current_value = self.pir_sensor.motion_detected
            except:
                current_value = False
        else:
            try:
                with open(f'/sys/class/gpio/gpio{self.pir_pin}/value', 'r') as f:
                    current_value = int(f.read().strip())
            except:
                current_value = 0
        
        return {
            'active': bool(current_value),
            'count': self.detection_count,
            'last_detection': self.last_detection
        }
    
    def cleanup(self):
        """Cleanup PIR resources"""
        if GPIO_METHOD == "gpiozero":
            self.pir_sensor.close()
        else:
            try:
                with open('/sys/class/gpio/unexport', 'w') as f:
                    f.write(str(self.pir_pin))
            except:
                pass

# ================= 기존 DoorBox 클래스 확장 =================

class DoorBoxInferenceSystem:
    def __init__(self):
        # 로깅 설정
        self._setup_logging()
        
        # 디렉토리 생성
        config.create_directories()
        
        # AWS S3 클라이언트 설정
        self._setup_aws_client()
        
        # 기본 설정
        self.rtsp_url = config.RTSP_URL
        self.serial_port = config.SERIAL_PORT
        self.serial_baudrate = config.SERIAL_BAUDRATE
        self.device_id = config.DEVICE_ID
        
        # 구성 요소 초기화
        self.cap = None
        self.ser = None
        self.running = False
        
        # ★ PIR, OLED, RGB LED 초기화
        self._init_hardware_components()
        
        # 모델들 로드
        self._load_models()
        
        # 비디오 레코더 초기화
        self._init_video_recorder()
        
        # S3 업로더 초기화
        self._init_s3_uploader()
        
        # 스레드 관리
        self.rtsp_thread = None
        self.serial_thread = None
        self.pir_monitor_thread = None
        
        # 프레임 버퍼
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 업로드 상태 초기화 (OLED 오류 해결)
        self.upload_status = {"active": False, "queue_size": 0}
        
        # PIR 기반 상태 관리
        self.system_state = "STANDBY"  # STANDBY, PIR_DETECTED, INFERENCE_ACTIVE
        self.pir_detection_time = None
        self.inference_timeout = 15.0  # 15초
        # 초록색 박스 감지 및 캡처 관리
        self.green_box_detected = False
        self.first_green_box_time = None
        self.last_capture_time = 0
        self.capture_delay = 2.0  # 첫 감지 후 2초 지연
        self.capture_interval = 5.0  # 5초 간격으로 캡처
        
        # OLED 분류 결과 표시 관리
        self.face_detection_results = None
        self.classification_display_start = None
        self.classification_display_duration = 3.0  # 3초 동안 표시
        
        # 기존 캡처 상태 관리
        self.capture_in_progress = False
        self.capture_lock = threading.Lock()
        self.last_successful_capture_time = 0
        self.capture_timer = None
        self.pending_captures = 0
        
        # 파일명 중복 방지를 위한 카운터
        self.filename_counter = {}
        self.filename_lock = threading.Lock()
        
        self.logger.info("DoorBox 추론 시스템 초기화 완료 (PIR 통합)")
    
    def _init_hardware_components(self):
        """PIR, OLED, RGB LED 초기화"""
        # RGB LED 초기화
        try:
            self.rgb = RGBController()
            self.rgb_available = True
            self.rgb.set_color_by_name("red")  # 초기 대기 상태는 빨간색
            self.logger.info("✅ RGB LED 초기화 완료")
        except Exception as e:
            self.logger.error(f"RGB LED 초기화 실패: {e}")
            self.rgb_available = False
        
        # OLED 디스플레이 초기화
        try:
            self.oled = OLEDDisplay()
            self.oled_available = True
            self.logger.info("✅ OLED 디스플레이 초기화 완료")
        except Exception as e:
            self.logger.error(f"OLED 초기화 실패: {e}")
            self.oled_available = False
        
        # PIR 센서 초기화
        try:
            self.pir = PIRSensor()
            self.pir.set_callback(self.on_pir_motion_detected)
            self.pir_available = True
            self.logger.info("✅ PIR 센서 초기화 완료")
        except Exception as e:
            self.logger.error(f"PIR 센서 초기화 실패: {e}")
            self.pir_available = False
    
    def on_pir_motion_detected(self):
        """PIR 센서 모션 감지 콜백"""
        self.logger.info("PIR 모션 감지 - 인퍼런스 모드 시작")
        self.system_state = "PIR_DETECTED"
        self.pir_detection_time = time.time()
        
        # RGB LED 흰색으로 변경 (PIR 감지됨)
        if self.rgb_available:
            self.rgb.set_color_by_name("white")
        
        # OLED 업데이트
        self._update_oled_display()
    
    def _update_oled_display(self):
        """OLED 디스플레이 업데이트 (분류 결과 3초 유지)"""
        if not self.oled_available:
            return
        
        try:
            current_time = time.time()
            pir_status = self.pir.get_status() if self.pir_available else None
            
            lines = [
                "DoorBox v2.0",
                f"Time: {datetime.now().strftime('%H:%M:%S')}",
                f"State: {self.system_state}",
            ]
            
            # PIR 상태 표시
            if pir_status:
                pir_state = "ACTIVE" if pir_status['active'] else "idle"
                lines.append(f"PIR: {pir_state} ({pir_status['count']})")
            else:
                lines.append("PIR: disabled")
            
            # 분류 결과가 있고 3초 이내인 경우 우선 표시
            if (self.face_detection_results and 
                self.classification_display_start and 
                current_time - self.classification_display_start < self.classification_display_duration):
                
                emotion = self.face_detection_results.get("emotion", "N/A")
                gender = self.face_detection_results.get("gender", "N/A")
                age_group = self.face_detection_results.get("age_group", "N/A")
                
                lines.append("=== RESULT ===")
                lines.append(f"Emotion: {emotion}")
                lines.append(f"Gender: {gender}")
                lines.append(f"Age: {age_group}")
                
            else:
                # 일반 상태 정보 표시
                if self.system_state == "STANDBY":
                    lines.append("Status: Waiting...")
                    
                elif self.system_state == "PIR_DETECTED":
                    if self.pir_detection_time:
                        elapsed = current_time - self.pir_detection_time
                        lines.append(f"PIR Timer: {elapsed:.1f}s")
                    
                elif self.system_state == "INFERENCE_ACTIVE":
                    if self.green_box_detected:
                        if self.first_green_box_time:
                            time_since_detection = current_time - self.first_green_box_time
                            if self.last_capture_time == 0 and time_since_detection < self.capture_delay:
                                remaining = self.capture_delay - time_since_detection
                                lines.append(f"Capture in {remaining:.1f}s")
                            else:
                                lines.append("Green Box: FOUND")
                    else:
                        lines.append("Green Box: Searching...")
                
                # S3 업로드 상태
                if self.upload_status["active"]:
                    lines.append(f"Upload: {self.upload_status['queue_size']} pending")
            
            # 최대 6줄까지만 표시
            self.oled.display_text(lines[:6])
            
        except Exception as e:
            self.logger.error(f"OLED 업데이트 오류: {e}")
    
    def _pir_monitor_worker(self):
        """PIR 센서 모니터링 스레드 (direct GPIO용)"""
        while self.running:
            if self.pir_available and GPIO_METHOD == "direct":
                self.pir.check_motion_direct()
            
            # 상태별 처리
            current_time = time.time()
            
            if self.system_state == "PIR_DETECTED":
                # PIR 감지 후 즉시 인퍼런스 활성 모드로 전환
                self.system_state = "INFERENCE_ACTIVE"
                self.logger.info("🔄 인퍼런스 활성 모드로 전환")
                
            elif self.system_state == "INFERENCE_ACTIVE":
                # 15초 타임아웃 체크
                if self.pir_detection_time and (current_time - self.pir_detection_time) > self.inference_timeout:
                    self.logger.info("⏰ 인퍼런스 타임아웃 - 대기 모드로 복귀")
                    self._return_to_standby()
            
            # OLED 주기적 업데이트
            self._update_oled_display()
            
            time.sleep(0.1)  # 100ms 간격으로 체크
    
    def _return_to_standby(self):
        """대기 모드로 복귀"""
        self.system_state = "STANDBY"
        self.pir_detection_time = None
        self.green_box_detected = False
        self.face_detection_results = None
        
        # RGB LED 빨간색으로 변경 (대기 상태)
        if self.rgb_available:
            self.rgb.set_color_by_name("red")
        
        # OLED 업데이트
        self._update_oled_display()
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 다른 라이브러리 로깅 레벨 조정
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    
    def _setup_aws_client(self):
        """AWS S3 클라이언트 설정"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        
        # S3 연결 테스트
        try:
            self.s3_client.head_bucket(Bucket=config.AWS_BUCKET_NAME)
            self.logger.info(f"✅ S3 버킷 연결 확인: {config.AWS_BUCKET_NAME}")
        except Exception as e:
            self.logger.error(f"S3 버킷 연결 실패: {e}")
    
    def _load_models(self):
        """3가지 분류 모델들 로드"""
        # 1. 감정 분류 모델 (EfficientNet-B0) - 320x320
        self.emotion_model = self._load_emotion_model()
        
        # 2. 연령대 분류 모델 (EfficientNet-B0) - 320x320
        self.age_model = self._load_efficientnet_model(config.AGE_MODEL_PATH, "연령대", num_classes=9)
        
        # 3. 성별 분류 모델 (MobileNetV3-Small) - 320x320
        self.gender_model = self._load_mobilenet_model(config.GENDER_MODEL_PATH, "성별")
        
        # 공통 전처리 (320x320로 통일)
        self.common_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_emotion_model(self):
        """EfficientNet-B0 모델 로드 (감정 분류용)"""
        try:
            model = efficientnet_b0(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 2)  # alert, non-alert
            )
            
            checkpoint = torch.load(config.EMOTION_MODEL_PATH, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()
            
            self.logger.info("✅ 감정 분류 모델 로드 완료")
            return model
        except Exception as e:
            self.logger.error(f"감정 모델 로드 실패: {e}")
            return None
    
    def _load_efficientnet_model(self, model_path, model_name, num_classes=9):
        """EfficientNet-B0 모델 로드 (연령대 분류용)"""
        try:
            if not os.path.exists(model_path):
                self.logger.warning(f"{model_name} 모델 파일 없음: {model_path}")
                return None
            
            # EfficientNet-B0 모델 생성
            model = efficientnet_b0(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )
            
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # state_dict 추출
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 모델에 로드
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            self.logger.info(f"✅ {model_name} 모델 로드 완료 (EfficientNet-B0, {num_classes}클래스)")
            return model
            
        except Exception as e:
            self.logger.error(f"{model_name} 모델 로드 실패: {e}")
            return None
    
    def _load_mobilenet_model(self, model_path, model_name):
        """MobileNetV3-Small 모델 로드 (성별 분류용)"""
        try:
            if not os.path.exists(model_path):
                self.logger.warning(f"{model_name} 모델 파일 없음: {model_path}")
                return None
            
            # MobileNetV3-Small 모델 생성
            model = mobilenet_v3_small(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(1024, 2)  # male, female
            )
            
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # state_dict 추출
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 모델에 로드
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            self.logger.info(f"✅ {model_name} 모델 로드 완료 (MobileNetV3-Small)")
            return model
            
        except Exception as e:
            self.logger.error(f"{model_name} 모델 로드 실패: {e}")
            return None
    
    def _classify_all_models(self, face_crop):
        """모든 모델로 분류 실행"""
        results = {
            "emotion": None,
            "emotion_confidence": 0.0,
            "gender": None,
            "gender_confidence": 0.0,
            "age_group": None,
            "age_confidence": 0.0
        }
        
        try:
            # 1. 감정 분류 (alert/non-alert로 변경)
            if self.emotion_model is not None:
                emotion, emotion_conf = self._classify_emotion(face_crop)
                results["emotion"] = emotion
                results["emotion_confidence"] = emotion_conf
            
            # 2. 성별 분류
            if self.gender_model is not None:
                gender, gender_conf = self._classify_gender(face_crop)
                results["gender"] = gender
                results["gender_confidence"] = gender_conf
            
            # 3. 연령대 분류
            if self.age_model is not None:
                age_group, age_conf = self._classify_age(face_crop)
                results["age_group"] = age_group
                results["age_confidence"] = age_conf
            
        except Exception as e:
            self.logger.error(f"분류 과정 오류: {e}")
        
        return results
    
    def _classify_emotion(self, face_crop):
        """감정 분류 (alert/non-alert)"""
        if self.emotion_model is None:
            return "unknown", 0.0
        
        try:
            input_tensor = self.common_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.emotion_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_classes = ["alert", "non-alert"]
                emotion = emotion_classes[predicted.item()]
                conf = confidence.item()
                
                return emotion, conf
        except Exception as e:
            self.logger.error(f"감정 분류 오류: {e}")
            return "unknown", 0.0

    def _classify_gender(self, face_crop):
        """성별 분류 (0=male, 1=female)"""
        if self.gender_model is None:
            return "unknown", 0.0
        
        try:
            input_tensor = self.common_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.gender_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                gender_classes = ["male", "female"]  # 0=male, 1=female
                gender = gender_classes[predicted.item()]
                conf = confidence.item()
                
                return gender, conf
        except Exception as e:
            self.logger.error(f"성별 분류 오류: {e}")
            return "unknown", 0.0
    
    def _classify_age(self, face_crop):
        """연령대 분류 (EfficientNet-B0, 9클래스: 0s, 10s, 20s, ..., 70s, over80s)"""
        if self.age_model is None:
            return "unknown", 0.0
        
        try:
            input_tensor = self.common_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.age_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # 9개 클래스: 0s, 10s, 20s, 30s, 40s, 50s, 60s, 70s, over80s
                age_classes = ["0s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "over80s"]
                age_group = age_classes[predicted.item()]
                conf = confidence.item()
                
                return age_group, conf
        except Exception as e:
            self.logger.error(f"연령대 분류 오류: {e}")
            return "unknown", 0.0
    
    def _init_video_recorder(self):
        """비디오 레코더 초기화"""
        self.frame_buffer = deque(maxlen=30 * config.VIDEO_CLIP_DURATION)  # 30fps * 5초
        self.recording = False
        self.buffer_thread = None
    
    def _start_video_buffering(self):
        """비디오 프레임 버퍼링 시작"""
        self.recording = True
        
        def buffer_frames():
            while self.recording and self.running:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        timestamp = time.time()
                        self.frame_buffer.append((self.latest_frame.copy(), timestamp))
                time.sleep(1/30)  # 30fps
        
        self.buffer_thread = threading.Thread(target=buffer_frames, daemon=True)
        self.buffer_thread.start()
    
    def _save_video_clip_improved(self, detection_time, output_path):
        """개선된 5초 비디오 클립 저장 (MP4 코덱만 사용)"""
        if not self.frame_buffer:
            self.logger.error("프레임 버퍼가 비어있음")
            return False
        
        try:
            # 감지 시점 기준으로 프레임 필터링
            clip_frames = []
            start_time = detection_time - config.PRE_BUFFER_DURATION
            end_time = detection_time + config.POST_BUFFER_DURATION
            
            self.logger.debug(f"비디오 클립 시간 범위: {start_time:.2f} ~ {end_time:.2f}")
            self.logger.debug(f"프레임 버퍼 크기: {len(self.frame_buffer)}개")
            
            for frame, timestamp in list(self.frame_buffer):
                if start_time <= timestamp <= end_time:
                    clip_frames.append(frame)
            
            self.logger.debug(f"클립용 프레임 수집: {len(clip_frames)}개")
            
            if len(clip_frames) < 10:
                self.logger.warning(f"클립 프레임 부족: {len(clip_frames)}개 (최소 10개 필요)")
                return False
            
            # 디렉토리 확인
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"비디오 디렉토리 생성: {output_dir}")
            
            # 비디오 파일로 저장 (MP4 코덱만 사용)
            height, width = clip_frames[0].shape[:2]
            
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
                
                if not out.isOpened():
                    self.logger.error("MP4 VideoWriter 열기 실패")
                    return False
                
                # 프레임 쓰기
                for frame in clip_frames:
                    out.write(frame)
                
                out.release()
                
                # 파일 생성 확인
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    self.logger.info("비디오 저장 성공 (MP4)")
                    return True
                else:
                    self.logger.error("MP4 파일 생성 실패")
                    return False
                    
            except Exception as e:
                self.logger.error(f"MP4 비디오 저장 오류: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"비디오 저장 오류: {e}")
            return False
    
    def _init_s3_uploader(self):
        """S3 업로더 초기화"""
        self.upload_queue = []
        self.upload_running = False
        self.upload_thread = None
    
    def _start_s3_uploader(self):
        """S3 업로드 스레드 시작"""
        def upload_worker():
            while self.upload_running:
                # 업로드 상태 업데이트
                self.upload_status["queue_size"] = len(self.upload_queue)
                self.upload_status["active"] = len(self.upload_queue) > 0
                
                # S3 업로드 시 별도의 LED 동작 없음
                
                self._process_upload_batch()
                time.sleep(config.UPLOAD_INTERVAL)
        
        self.upload_running = True
        self.upload_thread = threading.Thread(target=upload_worker, daemon=True)
        self.upload_thread.start()
        self.logger.info("S3 업로드 스레드 시작")
    
    def _queue_upload_data(self, frame_path, video_path, result_data, capture_timestamp=None):
        """S3 업로드 큐에 데이터 추가"""
        # ★ capture_timestamp가 제공되면 사용, 아니면 현재 시간
        if capture_timestamp is None:
            seoul_tz = pytz.timezone('Asia/Seoul')
            timestamp = datetime.now(seoul_tz)
        else:
            timestamp = capture_timestamp  # 캡처 시점의 정확한 시간 사용
        
        upload_item = {
            'timestamp': timestamp,
            'frame_path': frame_path,
            'video_path': video_path,
            'result_data': result_data,
            'uploaded': False,
            'retry_count': 0
        }
        
        self.upload_queue.append(upload_item)
        self.logger.info(f"S3 큐 추가 - 대기중: {len(self.upload_queue)}개")
    
    def _generate_s3_paths(self, timestamp):
        """S3 경로 생성 (중복 제거된 구조)"""
        # timestamp가 이미 서울 시간대면 그대로, 아니면 변환
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            # naive datetime이면 서울 시간대로 가정
            seoul_tz = pytz.timezone('Asia/Seoul')
            dt = seoul_tz.localize(timestamp)
        else:
            # timezone-aware datetime이면 서울 시간대로 변환
            seoul_tz = pytz.timezone('Asia/Seoul')
            dt = timestamp.astimezone(seoul_tz)
        
        # S3 저장 구조 (버킷 내부): home-1/cam-1/2025/08/23/...
        folder_name = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}_log"
        
        base_path = f"home-1/cam-1/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{folder_name}"
        file_prefix = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
        
        # JSON의 image_key는 전체 경로 (doorbox-data 포함)
        image_key_full_path = f"doorbox-data/home-1/cam-1/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{folder_name}/{file_prefix}_frame.jpg"
        
        return {
            'base_path': base_path,
            'frame_key': f"{base_path}/{file_prefix}_frame.jpg",
            'video_key': f"{base_path}/{file_prefix}_clip.mp4",
            'result_key': f"{base_path}/{file_prefix}_result.json",
            'image_key_full_path': image_key_full_path  # JSON용 전체 경로
        }
    
    def _upload_file_to_s3(self, local_path, s3_key, content_type):
        """단일 파일 S3 업로드"""
        try:
            with open(local_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=config.AWS_BUCKET_NAME,
                    Key=s3_key,
                    Body=f,
                    ContentType=content_type
                )
            return True
        except Exception as e:
            self.logger.error(f"S3 업로드 실패: {s3_key}, 오류: {e}")
            return False
    
    def _upload_json_to_s3(self, data, s3_key):
        """JSON 데이터 S3 업로드"""
        try:
            self.s3_client.put_object(
                Bucket=config.AWS_BUCKET_NAME,
                Key=s3_key,
                Body=json.dumps(data, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            self.logger.error(f"JSON 업로드 실패: {s3_key}, 오류: {e}")
            return False
    
    def _generate_unique_filename(self, base_timestamp_str):
        """중복 방지를 위한 고유 파일명 생성"""
        with self.filename_lock:
            if base_timestamp_str in self.filename_counter:
                self.filename_counter[base_timestamp_str] += 1
                counter = self.filename_counter[base_timestamp_str]
                return f"{base_timestamp_str}_{counter:02d}"
            else:
                self.filename_counter[base_timestamp_str] = 0
                return base_timestamp_str
    
    def _generate_s3_paths_with_custom_filename(self, timestamp, custom_filename):
        """커스텀 파일명을 사용한 S3 경로 생성"""
        # timestamp가 이미 서울 시간대면 그대로, 아니면 변환
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            seoul_tz = pytz.timezone('Asia/Seoul')
            dt = seoul_tz.localize(timestamp)
        else:
            seoul_tz = pytz.timezone('Asia/Seoul')
            dt = timestamp.astimezone(seoul_tz)
        
        # custom_filename 사용
        folder_name = f"{custom_filename}_log"
        
        base_path = f"home-1/cam-1/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{folder_name}"
        
        # JSON의 image_key는 전체 경로 (doorbox-data 포함)
        image_key_full_path = f"doorbox-data/home-1/cam-1/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{folder_name}/{custom_filename}_frame.jpg"
        
        return {
            'base_path': base_path,
            'frame_key': f"{base_path}/{custom_filename}_frame.jpg",
            'video_key': f"{base_path}/{custom_filename}_clip.mp4",
            'result_key': f"{base_path}/{custom_filename}_result.json",
            'image_key_full_path': image_key_full_path
        }
    
    def _process_upload_batch(self):
        """배치 업로드 처리"""
        if not self.upload_queue:
            # 업로드가 완료되면 현재 상태에 맞는 LED 색상으로 복귀
            self.upload_status["active"] = False
            self.upload_status["queue_size"] = 0
            
            if not self.upload_status["active"] and self.rgb_available:
                if self.system_state == "STANDBY":
                    self.rgb.set_color_by_name("red")
                elif self.system_state == "PIR_DETECTED" or self.system_state == "INFERENCE_ACTIVE":
                    self.rgb.set_color_by_name("white")
            return
        
        items_to_process = [item for item in self.upload_queue[:config.UPLOAD_BATCH_SIZE] 
                           if not item['uploaded'] and item['retry_count'] < 3]
        
        for item in items_to_process:
            try:
                s3_paths = self._generate_s3_paths(item['timestamp'])
                # 1. JSON 업로드
                if self._upload_json_to_s3(item['result_data'], s3_paths['result_key']):
                    self.logger.info(f"✅ JSON: {s3_paths['result_key']}")
                else:
                    item['retry_count'] += 1
                    continue
                
                # 2. 프레임 업로드
                if os.path.exists(item['frame_path']):
                    if self._upload_file_to_s3(item['frame_path'], s3_paths['frame_key'], 'image/jpeg'):
                        self.logger.info(f"✅ 프레임: {s3_paths['frame_key']}")
                        os.remove(item['frame_path'])
                    else:
                        item['retry_count'] += 1
                        continue
                
                # 3. 비디오 업로드
                if item['video_path'] and os.path.exists(item['video_path']):
                    if self._upload_file_to_s3(item['video_path'], s3_paths['video_key'], 'video/mp4'):
                        self.logger.info(f"✅ 비디오: {s3_paths['video_key']}")
                        os.remove(item['video_path'])
                    else:
                        item['retry_count'] += 1
                        continue
                
                item['uploaded'] = True
                self.logger.info(f"업로드 완료: {s3_paths['base_path']}")
                
            except Exception as e:
                self.logger.error(f"업로드 처리 오류: {e}")
                item['retry_count'] += 1
        
        # 완료/실패 아이템 제거
        self.upload_queue = [item for item in self.upload_queue 
                           if not item['uploaded'] and item['retry_count'] < 3]
        
        if items_to_process:
            self.logger.info(f"배치 완료 - 남은 큐: {len(self.upload_queue)}개")
    
    def _detect_green_boxes(self, frame):
        """초록색 박스 검출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 250 * 250:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
        
        return boxes
    
    def _expand_bbox(self, bbox, frame_shape, scale=1.5):
        """바운딩 박스 1.5배 확장"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        center_x = x + w // 2
        center_y = y + h // 2
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        
        new_x = min(new_x, frame_w - new_w)
        new_y = min(new_y, frame_h - new_h)
        new_w = min(new_w, frame_w - new_x)
        new_h = min(new_h, frame_h - new_y)
        
        return new_x, new_y, new_w, new_h
    
    def _save_detection_results_direct(self, frame, face_crop, classification_results, capture_timestamp, timestamp_str):
        """직접 저장 방식"""
        try:
            # 파일명 생성
            frame_filename = f"{timestamp_str}_frame.jpg"
            video_filename = f"{timestamp_str}_clip.mp4"
            result_filename = f"{timestamp_str}_result.json"
            
            # 저장 경로 생성 (절대경로 사용)
            frame_path = os.path.join(config.LOCAL_FRAMES_DIR, frame_filename)
            video_path = os.path.join(config.LOCAL_VIDEOS_DIR, video_filename)
            result_path = os.path.join(config.LOCAL_RESULTS_DIR, result_filename)
            
            # 디렉토리 생성 확인
            os.makedirs(config.LOCAL_FRAMES_DIR, exist_ok=True)
            os.makedirs(config.LOCAL_VIDEOS_DIR, exist_ok=True)
            os.makedirs(config.LOCAL_RESULTS_DIR, exist_ok=True)
            
            # 1. 프레임 저장 (직접 저장)
            success_frame = cv2.imwrite(frame_path, frame)
            if success_frame and os.path.exists(frame_path):
                file_size = os.path.getsize(frame_path)
                self.logger.info(f"프레임 저장 성공: {frame_filename} ({file_size} bytes)")
            else:
                self.logger.error(f"프레임 저장 실패: {frame_filename}")
                return False
            
            # 2. 얼굴 크롭 이미지도 별도 저장
            face_filename = f"{timestamp_str}_face.jpg"
            face_path = os.path.join(config.LOCAL_FRAMES_DIR, face_filename)
            cv2.imwrite(face_path, face_crop)
            
            # 3. 비디오 클립 저장 (현재 시점 기준)
            detection_time = time.time()
            clip_saved = self._save_video_clip_improved(detection_time, video_path)
            
            if clip_saved and os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                self.logger.info(f"비디오 저장 성공: {video_filename} ({file_size} bytes)")
            else:
                self.logger.warning(f"비디오 저장 실패: {video_filename}")
                video_path = None
            
            # 4. JSON 결과 데이터 생성
            s3_paths = self._generate_s3_paths_with_custom_filename(capture_timestamp, timestamp_str)
            result_data = {
                "day": capture_timestamp.strftime("%Y%m%d"),
                "time": capture_timestamp.strftime("%H:%M:%S"),
                "image_key": s3_paths['image_key_full_path'],
                "detection_results": {
                    "emotion": classification_results.get("emotion"),
                    "gender": classification_results.get("gender"),
                    "age_group": classification_results.get("age_group"),
                }
            }
            
            # 5. JSON 파일 저장
            try:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                if os.path.exists(result_path):
                    file_size = os.path.getsize(result_path)
                    self.logger.info(f"결과 JSON 저장 성공: {result_filename} ({file_size} bytes)")
                else:
                    self.logger.error(f"JSON 저장 실패: {result_filename}")
            except Exception as e:
                self.logger.error(f"JSON 저장 오류: {e}")
            
            # 6. S3 업로드 큐에 추가
            self._queue_upload_data(frame_path, video_path, result_data, capture_timestamp)
            
            # 7. 상세 로그 출력
            emotion = classification_results.get("emotion", "unknown")
            emotion_conf = classification_results.get("emotion_confidence", 0.0)
            gender = classification_results.get("gender", "unknown")
            gender_conf = classification_results.get("gender_confidence", 0.0)
            age_group = classification_results.get("age_group", "unknown")
            age_conf = classification_results.get("age_confidence", 0.0)
            
            # 깔끔한 로그 출력
            self.logger.info("=" * 50)
            self.logger.info(f"프레임 파일: {frame_filename}")
            self.logger.info(f"캡처 시간: {capture_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=== 분류 결과 ===")
            self.logger.info(f"   감정: {emotion} (신뢰도: {emotion_conf:.3f})")
            self.logger.info(f"   성별: {gender} (신뢰도: {gender_conf:.3f})")
            self.logger.info(f"   연령대: {age_group} (신뢰도: {age_conf:.3f})")
            self.logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"결과 저장 오류: {e}")
            return False
    
    def _rtsp_capture_worker(self):
        """RTSP 캡처 스레드 (최적화)"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            self.logger.error("RTSP 연결 실패")
            return
        
        # RTSP 스트림 최적화 설정
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화 (지연 시간 감소)
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # FPS 설정
        
        # 초기 몇 프레임은 버림 (연결 안정화)
        for _ in range(5):
            self.cap.read()
        
        self.logger.info("RTSP 스트림 시작 (최적화됨)")
        
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                frame_count += 1
                
            else:
                self.logger.warning("프레임 읽기 실패")
                time.sleep(0.01)  # 짧은 대기
            
            # CPU 사용률 최적화
            time.sleep(0.001)  # 1ms 대기 (기존 0.1초에서 단축)
        
        self.cap.release()
        self.logger.info("RTSP 캡처 종료")
    
    def _serial_reader_worker(self):
        """시리얼 통신 스레드 (최적화)"""
        try:
            self.ser = serial.Serial(self.serial_port, self.serial_baudrate, timeout=0.1)  # 타임아웃 단축
            self.logger.info(f"시리얼 연결 성공: {self.serial_port}")
        except Exception as e:
            self.logger.error(f"시리얼 연결 실패: {e}")
            return
        
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        # PIR 기반 시스템에서는 시리얼 데이터 대신 프레임 기반 처리만 수행
                        # YOLO 결과는 참고만 하고, 실제 처리는 PIR 트리거 기반으로 수행
                        if "[print_yolo_result]" in line and "[AI coordinate]" in line:
                            # 로그만 출력하고 별도 처리 안함
                            self.logger.debug(f"YOLO 결과 수신: {line}")
                
                time.sleep(0.001)  # 1ms 대기 (기존 0.1초에서 단축)
                
            except Exception as e:
                self.logger.error(f"시리얼 읽기 오류: {e}")
                time.sleep(0.1)
        
        if self.ser:
            self.ser.close()
        self.logger.info("시리얼 통신 종료")
    
    def _process_inference_frame(self):
        """PIR 감지 후 인퍼런스 프레임 처리 (초록색 박스 기반 캡처 타이밍 개선)"""
        if self.system_state != "INFERENCE_ACTIVE":
            return
        
        current_time = time.time()
        
        # 세션 타임아웃 체크
        if current_time - self.pir_detection_time > config.DETECTION_TIMEOUT:
            self._return_to_standby()
            return
        
        # 캡처 시점의 정확한 타임스탬프 생성
        seoul_tz = pytz.timezone('Asia/Seoul')
        capture_timestamp = datetime.now(seoul_tz)
        
        with self.frame_lock:
            if self.latest_frame is None:
                self.logger.warning("사용 가능한 프레임이 없음")
                return
            current_frame = self.latest_frame.copy()
        
        # 초록색 박스 검출
        boxes = self._detect_green_boxes(current_frame)
        
        if not boxes:
            # 초록색 박스가 없으면 상태 초기화
            if self.green_box_detected:
                self.green_box_detected = False
                self.first_green_box_time = None
                self.logger.debug("초록색 박스 사라짐 - 상태 초기화")
            return
        
        # 초록색 박스 감지됨
        if not self.green_box_detected:
            # 첫 번째 초록색 박스 감지
            self.green_box_detected = True
            self.first_green_box_time = current_time
            self.logger.info("초록색 박스 첫 감지 - 2초 후 캡처 시작 예정")
            
            # RGB LED 보라색으로 점멸
            if self.rgb_available:
                self.rgb.blink_purple(times=3, interval=0.3)
                self.rgb.set_color_by_name("white")
            
            return
        
        # 초록색 박스가 지속적으로 감지되는 상태에서 캡처 타이밍 결정
        time_since_first_detection = current_time - self.first_green_box_time
        
        should_capture = False
        
        if self.last_capture_time == 0:
            # 첫 번째 캡처: 첫 감지 후 2초 지연
            if time_since_first_detection >= self.capture_delay:
                should_capture = True
        else:
            # 후속 캡처: 마지막 캡처로부터 5초 간격
            if current_time - self.last_capture_time >= self.capture_interval:
                should_capture = True
        
        if not should_capture:
            return
        
        # 캡처 및 분류 실행
        self.logger.info("초록색 박스 지속 감지 - 얼굴 분류 실행")
        
        # 가장 큰 박스 선택
        largest_box = max(boxes, key=lambda box: box[2] * box[3])
        
        # 박스 확장
        expanded_box = self._expand_bbox(largest_box, current_frame.shape)
        x, y, w, h = expanded_box
        
        # 얼굴 영역 크롭
        face_crop = current_frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            self.logger.warning("크롭된 얼굴 영역이 비어있음")
            return
        
        # 고유 파일명 생성 (중복 방지)
        base_timestamp_str = capture_timestamp.strftime("%Y%m%d_%H%M%S")
        unique_timestamp_str = self._generate_unique_filename(base_timestamp_str)
        
        # 모든 모델로 분류 실행
        classification_results = self._classify_all_models(face_crop)
        
        # 결과 저장 (직접 저장 방식 사용)
        success = self._save_detection_results_direct(
            current_frame, 
            face_crop,
            classification_results, 
            capture_timestamp, 
            unique_timestamp_str
        )
        
        if success:
            self.last_capture_time = current_time
            # OLED 표시용 결과 저장 및 표시 시작 시간 설정
            self.face_detection_results = classification_results
            self.classification_display_start = current_time
    
    def _inference_loop_worker(self):
        """인퍼런스 루프 스레드"""
        while self.running:
            if self.system_state == "INFERENCE_ACTIVE":
                try:
                    self._process_inference_frame()
                    time.sleep(1.0)  # 1초 간격으로 프레임 처리
                except Exception as e:
                    self.logger.error(f"인퍼런스 처리 오류: {e}")
            else:
                time.sleep(0.1)  # 비활성 상태에서는 짧은 대기
    
    # 추가 유틸리티 함수들
    def search_logs_by_filename(self, filename_pattern):
        """파일명 패턴으로 로그 검색 (디버깅용)"""
        log_file = config.LOG_FILE
        results = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            in_classification_block = False
            current_block = []
            
            for line in lines:
                if filename_pattern in line and "프레임 파일:" in line:
                    in_classification_block = True
                    current_block = [line.strip()]
                elif in_classification_block:
                    current_block.append(line.strip())
                    if "=" * 50 in line:  # 블록 종료
                        results.append('\n'.join(current_block))
                        in_classification_block = False
                        current_block = []
        
        except Exception as e:
            self.logger.error(f"로그 검색 오류: {e}")
        
        return results
    
    def debug_classification_by_file(self, frame_filename):
        """특정 프레임 파일의 분류 결과 조회"""
        # 파일명에서 확장자 제거
        base_name = frame_filename.replace('_frame.jpg', '').replace('.jpg', '')
        
        # 로그에서 해당 파일의 분류 결과 검색
        results = self.search_logs_by_filename(base_name)
        
        if results:
            print(f"\n🔍 {frame_filename}의 분류 결과:")
            for result in results:
                print(result)
        else:
            print(f"❌ {frame_filename}에 대한 분류 결과를 찾을 수 없습니다.")
        
        return results
    
    def start(self):
        """시스템 시작"""
        self.running = True
        
        # 하드웨어 상태 표시
        self.logger.info("=" * 60)
        self.logger.info("🚪 DoorBox PIR 통합 시스템")
        self.logger.info("=" * 60)
        self.logger.info(f"💡 RGB LED: {'활성' if self.rgb_available else '비활성'}")
        self.logger.info(f"📺 OLED Display: {'활성' if self.oled_available else '비활성'}")
        self.logger.info(f"🚶 PIR Sensor: {'활성' if self.pir_available else '비활성'}")
        self.logger.info(f"🔧 GPIO Method: {GPIO_METHOD}")
        self.logger.info("=" * 60)
        
        # 시스템 초기 상태 설정
        self._return_to_standby()
        
        # 비디오 버퍼링 시작
        self._start_video_buffering()
        
        # S3 업로더 시작
        self._start_s3_uploader()
        
        # RTSP 캡처 스레드 시작
        self.rtsp_thread = threading.Thread(target=self._rtsp_capture_worker, daemon=True)
        self.rtsp_thread.start()
        
        # 시리얼 리더 스레드 시작
        self.serial_thread = threading.Thread(target=self._serial_reader_worker, daemon=True)
        self.serial_thread.start()
        
        # PIR 모니터링 스레드 시작 (direct GPIO용)
        self.pir_monitor_thread = threading.Thread(target=self._pir_monitor_worker, daemon=True)
        self.pir_monitor_thread.start()
        
        # 인퍼런스 루프 스레드 시작
        self.inference_thread = threading.Thread(target=self._inference_loop_worker, daemon=True)
        self.inference_thread.start()
        
        self.logger.info("✅ DoorBox PIR 통합 시스템 시작됨")
        
        # 메인 루프
        try:
            while self.running:
                # 주기적으로 OLED 업데이트
                self._update_oled_display()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("키보드 인터럽트 감지")
            self.stop()
    
    def stop(self):
        """시스템 종료"""
        self.logger.info("🛑 시스템 종료 중...")
        
        self.running = False
        
        # 비디오 레코더 종료
        self.recording = False
        if self.buffer_thread:
            self.buffer_thread.join(timeout=2)
        
        # S3 업로더 종료 (남은 큐 처리)
        if self.upload_queue:
            self.logger.info("남은 업로드 처리 중...")
            self._process_upload_batch()
        
        self.upload_running = False
        if self.upload_thread:
            self.upload_thread.join(timeout=5)
        
        # 스레드 종료 대기
        if self.rtsp_thread:
            self.rtsp_thread.join(timeout=3)
        if self.serial_thread:
            self.serial_thread.join(timeout=3)
        if self.pir_monitor_thread:
            self.pir_monitor_thread.join(timeout=3)
        if hasattr(self, 'inference_thread') and self.inference_thread:
            self.inference_thread.join(timeout=3)
        
        # 하드웨어 정리
        if self.rgb_available:
            self.rgb.cleanup()
        
        if self.oled_available:
            self.oled.clear()
        
        if self.pir_available:
            self.pir.cleanup()
        
        self.logger.info("✅ DoorBox PIR 통합 시스템 종료 완료")

def main():
    """메인 실행 함수"""
    print("🚪 DoorBox PIR 통합 시스템")
    print("하드웨어 연결:")
    print("  RGB LED: R=Pin12, G=Pin16, B=Pin18, GND=Pin9")
    print("  OLED: SDA=Pin3, SCL=Pin5, VCC=Pin1, GND=Pin6") 
    print("  PIR: VCC=Pin4, OUT=Pin8, GND=Pin9")
    print("  CatchCAM: USB + UART")
    print()
    
    doorbox = None
    
    try:
        doorbox = DoorBoxInferenceSystem()
        doorbox.start()
    except Exception as e:
        logging.error(f"시스템 오류: {e}")
    finally:
        if doorbox:
            doorbox.stop()

if __name__ == "__main__":
    main()
