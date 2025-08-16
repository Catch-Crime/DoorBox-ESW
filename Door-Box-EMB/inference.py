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
import torchvision  # 이 import가 warning을 미리 발생시켜서 나중에 안 나옴

# 설정 파일 import
import config

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
        
        # 모델들 로드
        self._load_all_models()
        
        # 비디오 레코더 초기화
        self._init_video_recorder()
        
        # S3 업로더 초기화
        self._init_s3_uploader()
        
        # 스레드 관리
        self.rtsp_thread = None
        self.serial_thread = None
        
        # 프레임 버퍼
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 감지 상태 관리
        self.detection_active = False
        self.first_detection_time = None
        self.last_capture_time = None
        self.detection_session_id = None
        
        self.logger.info("DoorBox 추론 시스템 초기화 완료")
    
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
    
    def _load_all_models(self):
        """모든 분류 모델 로드"""
        # 1. 감정 분류 모델 (EfficientNet-B0)
        self.emotion_model = self._load_emotion_model()
        self.emotion_transform = self._get_emotion_transform()
        
        # 2. 악세서리(마스크) 분류 모델 (GhostNet)
        self.accessory_model = self._load_ghostnet_model(config.ACCESSORY_MODEL_PATH, "악세서리")
        
        # 3. 연령대 분류 모델 (EfficientNet-B0)
        self.age_model = self._load_efficientnet_model(config.AGE_MODEL_PATH, "연령대", num_classes=9)
        
        # 4. 성별 분류 모델 (MobileNetV3-Small)
        self.gender_model = self._load_mobilenet_model(config.GENDER_MODEL_PATH, "성별")
        
        # 공통 전처리 (224x224)
        self.common_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_emotion_model(self):
        """감정 분류 모델 로드 (EfficientNet-B0)"""
        try:
            model = efficientnet_b0(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 2)  # negative, non-negative
            )
            
            checkpoint = torch.load(config.EMOTION_MODEL_PATH, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()
            
            self.logger.info("✅ 감정 분류 모델 로드 완료")
            return model
        except Exception as e:
            self.logger.error(f"감정 모델 로드 실패: {e}")
            return None
    
    def _load_ghostnet_model(self, model_path, model_name):
        """GhostNet 모델 로드 (악세서리/마스크 분류용)"""
        try:
            if not os.path.exists(model_path):
                self.logger.warning(f"{model_name} 모델 파일 없음: {model_path}")
                return None
            
            # 체크포인트 먼저 로드해서 구조 확인
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # state_dict 추출
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # classifier 크기 확인
            classifier_weight_shape = None
            for key in state_dict.keys():
                if 'classifier.weight' in key:
                    classifier_weight_shape = state_dict[key].shape
                    break
            
            if classifier_weight_shape is not None:
                num_classes, feature_dim = classifier_weight_shape
                self.logger.info(f"{model_name} 모델 구조: {num_classes}클래스, {feature_dim}차원")
                
                # 커스텀 GhostNet 모델 생성 (feature_dim에 맞춰)
                class CustomGhostNet(nn.Module):
                    def __init__(self, num_classes=2, feature_dim=960):
                        super().__init__()
                        # GhostNet backbone (feature extractor only)
                        self.backbone = timm.create_model('ghostnet_100', pretrained=False, num_classes=0)
                        
                        # backbone 출력 차원 확인 및 조정
                        backbone_dim = self.backbone.num_features
                        
                        # feature_dim에 맞춰 projection layer 추가
                        if backbone_dim != feature_dim:
                            self.projection = nn.Linear(backbone_dim, feature_dim)
                        else:
                            self.projection = nn.Identity()
                        
                        # classifier
                        self.classifier = nn.Linear(feature_dim, num_classes)
                    
                    def forward(self, x):
                        features = self.backbone(x)
                        features = self.projection(features)
                        return self.classifier(features)
                
                model = CustomGhostNet(num_classes, feature_dim)
                
            else:
                # 기본 모델 생성
                model = timm.create_model('ghostnet_100', pretrained=False, num_classes=2)
            
            # 모델에 로드 (strict=False로 호환성 확보)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                self.logger.warning(f"누락된 키: {len(missing_keys)}개")
            if unexpected_keys:
                self.logger.warning(f"예상치 못한 키: {len(unexpected_keys)}개")
            
            model.eval()
            
            self.logger.info(f"✅ {model_name} 모델 로드 완료 (Custom GhostNet)")
            return model
            
        except Exception as e:
            self.logger.error(f"{model_name} 모델 로드 실패: {e}")
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
    
    def _get_emotion_transform(self):
        """감정 분류용 이미지 전처리"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _classify_all_models(self, face_crop):
        """모든 모델로 분류 실행"""
        results = {
            "emotion": None,
            "emotion_confidence": 0.0,
            "has_mask": None,
            "mask_confidence": 0.0,
            "gender": None,
            "gender_confidence": 0.0,
            "age_group": None,
            "age_confidence": 0.0
        }
        
        try:
            # 1. 악세서리(마스크) 분류 - 우선 실행
            if self.accessory_model is not None:
                mask_result, mask_conf = self._classify_accessory(face_crop)
                results["has_mask"] = mask_result
                results["mask_confidence"] = mask_conf
                
                # 마스크 착용시 다른 분류 건너뛰기
                if mask_result:
                    self.logger.info("마스크 착용 감지 - 다른 분류 생략")
                    return results
            
            # 2. 감정 분류
            if self.emotion_model is not None:
                emotion, emotion_conf = self._classify_emotion(face_crop)
                results["emotion"] = emotion
                results["emotion_confidence"] = emotion_conf
            
            # 3. 성별 분류
            if self.gender_model is not None:
                gender, gender_conf = self._classify_gender(face_crop)
                results["gender"] = gender
                results["gender_confidence"] = gender_conf
            
            # 4. 연령대 분류
            if self.age_model is not None:
                age_group, age_conf = self._classify_age(face_crop)
                results["age_group"] = age_group
                results["age_confidence"] = age_conf
            
        except Exception as e:
            self.logger.error(f"분류 과정 오류: {e}")
        
        return results
    
    def _classify_emotion(self, face_crop):
        """감정 분류"""
        if self.emotion_model is None:
            return "unknown", 0.0
        
        try:
            input_tensor = self.emotion_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.emotion_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_classes = ["negative", "non-negative"]
                emotion = emotion_classes[predicted.item()]
                conf = confidence.item()
                
                return emotion, conf
        except Exception as e:
            self.logger.error(f"감정 분류 오류: {e}")
            return "unknown", 0.0
    
    def _classify_accessory(self, face_crop):
        """악세서리(마스크) 분류"""
        if self.accessory_model is None:
            return None, 0.0
        
        try:
            input_tensor = self.common_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                # backbone으로 feature 추출
                if hasattr(self.accessory_model, 'backbone'):
                    features = self.accessory_model.backbone(input_tensor)
                    
                    # 차원 맞춤: 1280 → 960
                    if features.shape[1] == 1280:
                        # 간단한 차원 축소 (첫 960개 차원만 사용)
                        features = features[:, :960]
                    
                    # classifier 적용
                    outputs = self.accessory_model.classifier(features)
                else:
                    # 일반적인 forward
                    outputs = self.accessory_model(input_tensor)
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # 0: 마스크 없음, 1: 마스크 있음
                has_mask = bool(predicted.item())
                conf = confidence.item()
                
                # 임계값 적용 (신뢰도가 0.7 이상일 때만 마스크 착용으로 판단)
                if conf < 0.7:
                    has_mask = False
                
                return has_mask, conf
        except Exception as e:
            self.logger.error(f"악세서리 분류 오류: {e}")
            # 에러 발생시 기본값 반환
            return False, 0.0
    
    def _classify_gender(self, face_crop):
        """성별 분류"""
        if self.gender_model is None:
            return "unknown", 0.0
        
        try:
            input_tensor = self.common_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.gender_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                gender_classes = ["male", "female"]
                gender = gender_classes[predicted.item()]
                conf = confidence.item()
                
                return gender, conf
        except Exception as e:
            self.logger.error(f"성별 분류 오류: {e}")
            return "unknown", 0.0
    
    def _classify_age(self, face_crop):
        """연령대 분류 (EfficientNet-B0, 9클래스)"""
        if self.age_model is None:
            return "unknown", 0.0
        
        try:
            input_tensor = self.common_transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.age_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # 9개 클래스로 확장 (실제 클래스명은 모델에 따라 조정)
                age_classes = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
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
    
    def _save_video_clip(self, detection_time, output_path):
        """5초 비디오 클립 저장"""
        if not self.frame_buffer:
            return False
        
        try:
            # 감지 시점 기준으로 프레임 필터링
            clip_frames = []
            start_time = detection_time - config.PRE_BUFFER_DURATION
            end_time = detection_time + config.POST_BUFFER_DURATION
            
            for frame, timestamp in self.frame_buffer:
                if start_time <= timestamp <= end_time:
                    clip_frames.append(frame)
            
            if len(clip_frames) < 10:
                return False
            
            # 비디오 파일로 저장
            height, width = clip_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            for frame in clip_frames:
                out.write(frame)
            
            out.release()
            return True
            
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
                self._process_upload_batch()
                time.sleep(config.UPLOAD_INTERVAL)
        
        self.upload_running = True
        self.upload_thread = threading.Thread(target=upload_worker, daemon=True)
        self.upload_thread.start()
        self.logger.info("S3 업로드 스레드 시작")
    
    def _queue_upload_data(self, frame_path, video_path, result_data):
        """S3 업로드 큐에 데이터 추가"""
        # 서울 시간대로 타임스탬프 생성
        seoul_tz = pytz.timezone('Asia/Seoul')
        timestamp = datetime.now(seoul_tz)
        
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
        """S3 경로 생성 (서울 시간대 기준)"""
        # timestamp가 이미 서울 시간대면 그대로, 아니면 변환
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            # naive datetime이면 서울 시간대로 가정
            seoul_tz = pytz.timezone('Asia/Seoul')
            dt = seoul_tz.localize(timestamp)
        else:
            # timezone-aware datetime이면 서울 시간대로 변환
            seoul_tz = pytz.timezone('Asia/Seoul')
            dt = timestamp.astimezone(seoul_tz)
        
        folder_name = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}_log"
        
        base_path = f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{folder_name}"
        file_prefix = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
        
        return {
            'base_path': base_path,
            'frame_key': f"{base_path}/{file_prefix}_frame.jpg",
            'video_key': f"{base_path}/{file_prefix}_clip.mp4",
            'result_key': f"{base_path}/{file_prefix}_result.json"
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
    
    def _process_upload_batch(self):
        """배치 업로드 처리"""
        if not self.upload_queue:
            return
        
        items_to_process = [item for item in self.upload_queue[:config.UPLOAD_BATCH_SIZE] 
                           if not item['uploaded'] and item['retry_count'] < 3]
        
        for item in items_to_process:
            try:
                s3_paths = self._generate_s3_paths(item['timestamp'])
                
                # 이미지 키 설정
                item['result_data']['image_key'] = f"{config.AWS_BUCKET_NAME}/{s3_paths['frame_key']}"
                
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
    
    def _save_detection_results(self, frame, classification_results):
        """감지 결과 저장 및 S3 큐 추가"""
        try:
            # 서울 시간대 설정
            seoul_tz = pytz.timezone('Asia/Seoul')
            timestamp = datetime.now(seoul_tz)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            frame_filename = f"{timestamp_str}_frame.jpg"
            video_filename = f"{timestamp_str}_clip.mp4"
            result_filename = f"{timestamp_str}_result.json"
            
            frame_path = os.path.join(config.LOCAL_FRAMES_DIR, frame_filename)
            video_path = os.path.join(config.LOCAL_VIDEOS_DIR, video_filename)
            result_path = os.path.join(config.LOCAL_RESULTS_DIR, result_filename)
            
            # 1. 프레임 저장
            cv2.imwrite(frame_path, frame)
            self.logger.info(f"프레임 저장: {frame_filename}")
            
            # 2. 5초 클립 저장
            detection_time = time.time()
            clip_saved = self._save_video_clip(detection_time, video_path)
            
            if clip_saved:
                self.logger.info(f"비디오 저장: {video_filename}")
            else:
                self.logger.warning("비디오 저장 실패")
                video_path = None
            
            # 3. JSON 데이터 생성 (서울 시간대)
            result_data = {
                "day": timestamp.strftime("%Y%m%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "detection_results": {
                    "emotion": classification_results.get("emotion"),
                    "confidence": round(classification_results.get("emotion_confidence", 0.0), 3),
                    "has_mask": classification_results.get("has_mask"),
                    "gender": classification_results.get("gender"),
                    "age_group": classification_results.get("age_group")
                },
                "image_key": ""  # S3 업로드시 설정
            }
            
            # 4. 로컬에 JSON 파일 저장
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"결과 JSON 저장: {result_filename}")
            
            # 5. S3 업로드 큐에 추가
            self._queue_upload_data(frame_path, video_path, result_data)
            
            # 6. 상세 로그 출력 (모든 분류 결과 표시)
            emotion = classification_results.get("emotion", "unknown")
            emotion_conf = classification_results.get("emotion_confidence", 0.0)
            has_mask = classification_results.get("has_mask")
            mask_conf = classification_results.get("mask_confidence", 0.0)
            gender = classification_results.get("gender", "unknown")
            gender_conf = classification_results.get("gender_confidence", 0.0)
            age_group = classification_results.get("age_group", "unknown")
            age_conf = classification_results.get("age_confidence", 0.0)
            
            # 마스크 상태 텍스트
            if has_mask is True:
                mask_text = f"마스크 착용({mask_conf:.3f})"
            elif has_mask is False:
                mask_text = f"마스크 없음({mask_conf:.3f})"
            else:
                mask_text = "마스크 미판별"
            
            self.logger.info("=== 분류 결과 상세 ===")
            self.logger.info(f"   감정: {emotion} (신뢰도: {emotion_conf:.3f})")
            self.logger.info(f"   마스크: {mask_text}")
            self.logger.info(f"   성별: {gender} (신뢰도: {gender_conf:.3f})")
            self.logger.info(f"   연령대: {age_group} (신뢰도: {age_conf:.3f})")
            self.logger.info("========================")
            
        except Exception as e:
            self.logger.error(f"결과 저장 오류: {e}")
    
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
                        # YOLO 감지 결과만 처리 (로그 출력 안함)
                        if "[print_yolo_result]" in line and "[AI coordinate]" in line:
                            self._process_yolo_detection(line)
                        # 다른 시리얼 메시지는 무시
                
                time.sleep(0.001)  # 1ms 대기 (기존 0.1초에서 단축)
                
            except Exception as e:
                self.logger.error(f"시리얼 읽기 오류: {e}")
                time.sleep(0.1)
        
        if self.ser:
            self.ser.close()
        self.logger.info("시리얼 통신 종료")
    
    def _process_yolo_detection(self, yolo_line):
        """YOLO 감지 결과 처리"""
        current_time = time.time()
        
        if not self.detection_active:
            # 새로운 감지 세션 시작
            self.detection_active = True
            self.first_detection_time = current_time
            self.detection_session_id = int(current_time)
            self.last_capture_time = None
            self.logger.info(f"새로운 감지 세션 시작 (ID: {self.detection_session_id})")
            
            # 첫 감지 후 즉시 또는 지연 후 캡처 시작
            if config.DETECTION_DELAY > 0:
                threading.Timer(config.DETECTION_DELAY, self._start_periodic_capture).start()
                self.logger.info(f"{config.DETECTION_DELAY}초 후 캡처 시작 예정")
            else:
                self._start_periodic_capture()
        
        # 감지 세션 타임아웃 체크
        if current_time - self.first_detection_time > config.DETECTION_TIMEOUT:
            self._end_detection_session()
    
    def _start_periodic_capture(self):
        """주기적 캡처 시작"""
        if self.detection_active:
            self.logger.info("캡처 및 분류 시작")
            self._capture_and_process()
            
            # 다음 캡처 스케줄링
            if self.detection_active:
                threading.Timer(config.CAPTURE_INTERVAL, self._start_periodic_capture).start()
    
    def _capture_and_process(self):
        """캡처 및 처리 실행"""
        current_time = time.time()
        
        # 감지 세션이 활성화되어 있는지 확인
        if not self.detection_active:
            return
        
        # 세션 타임아웃 체크
        if current_time - self.first_detection_time > config.DETECTION_TIMEOUT:
            self._end_detection_session()
            return
        
        with self.frame_lock:
            if self.latest_frame is None:
                self.logger.warning("사용 가능한 프레임이 없음")
                return
            current_frame = self.latest_frame.copy()
        
        # 초록색 박스 검출
        boxes = self._detect_green_boxes(current_frame)
        
        if not boxes:
            self.logger.info("초록색 박스 없음 - 캡처 건너뜀")
            return
        
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
        
        # 모든 모델로 분류 실행
        classification_results = self._classify_all_models(face_crop)
        
        # 결과 저장 및 S3 업로드 큐 추가
        self._save_detection_results(current_frame, classification_results)
        
        self.last_capture_time = current_time
    
    def _end_detection_session(self):
        """감지 세션 종료"""
        if self.detection_active:
            self.logger.info(f"감지 세션 종료 (ID: {self.detection_session_id})")
            self.detection_active = False
            self.first_detection_time = None
            self.last_capture_time = None
            self.detection_session_id = None
    
    def start(self):
        """시스템 시작"""
        self.running = True
        
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
        
        self.logger.info("DoorBox 시스템 시작됨")
        
        # 메인 루프
        try:
            while self.running:
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
        
        self.logger.info("✅ DoorBox 시스템 종료 완료")

def main():
    """메인 실행 함수"""
    doorbox = DoorBoxInferenceSystem()
    
    try:
        doorbox.start()
    except Exception as e:
        logging.error(f"시스템 오류: {e}")
    finally:
        doorbox.stop()

if __name__ == "__main__":
    main()
