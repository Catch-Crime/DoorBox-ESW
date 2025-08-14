#!/usr/bin/env python3
"""
Door-Box Inference System (Raspberry Pi + CatchCAM)
초록색 박스 자동 인식 방식: 좌표 변환 없이 그려진 박스를 직접 검출, 1.5배 확장 ver
"""

import cv2
import torch
import serial
import numpy as np
import json
import time
import threading
from datetime import datetime
import argparse
from pathlib import Path

class DoorBoxInference:
    def __init__(self, 
                 rtsp_url="rtsp://10.0.0.156/live1.sdp",
                 serial_port="/dev/ttyUSB0",
                 serial_baudrate=115200,
                 emotion_model_path="models/emotion2.pth",
                 output_dir="output",
                 save_visualization=True):
        
        self.rtsp_url = rtsp_url
        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate
        self.emotion_model_path = emotion_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_visualization = save_visualization
        
        # 최신 프레임 저장
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 모델 초기화
        self.emotion_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 상태 플래그
        self.running = False
        
        # CatchCAM 설정
        self.frame_width = 1920
        self.frame_height = 1080
        
        # 디버그 모드
        self.debug_mode = True

    def detect_green_boxes(self, frame):
        """프레임에서 초록색 박스들을 검출"""
        try:
            # HSV 색공간으로 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 초록색 범위 정의
            green_ranges = [
                ((40, 100, 100), (80, 255, 255)),  # 밝은 초록
                ((35, 150, 50), (85, 255, 255)),   # 진한 초록
                ((45, 50, 50), (75, 255, 255)),    # 연한 초록
            ]
            
            all_boxes = []
            
            for i, (lower, upper) in enumerate(green_ranges):
                lower_green = np.array(lower)
                upper_green = np.array(upper)
                
                # 초록색 마스크 생성
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # 노이즈 제거
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # 윤곽선 검출
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 윤곽선을 직사각형으로 근사
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 박스 크기 필터링 (최소 250x250으로 변경)
                    if 250 <= w <= 800 and 250 <= h <= 800:
                        aspect_ratio = w / h
                        if 0.3 <= aspect_ratio <= 2.0:
                            area = w * h
                            perimeter = cv2.arcLength(contour, True)
                            
                            if perimeter > 0:
                                rectangularity = (4 * np.pi * area) / (perimeter * perimeter)
                                if rectangularity > 0.2:
                                    all_boxes.append({
                                        'bbox': [x, y, x+w, y+h],
                                        'area': area,
                                        'aspect_ratio': aspect_ratio,
                                        'rectangularity': rectangularity,
                                        'range_index': i
                                    })
            
            # 중복 제거
            filtered_boxes = self.remove_overlapping_boxes(all_boxes)
            
            if self.debug_mode and filtered_boxes:
                print(f"\n🟢 초록색 박스 검출: {len(filtered_boxes)}개")
                for i, box in enumerate(filtered_boxes):
                    bbox = box['bbox']
                    print(f"   박스{i+1}: {bbox} (크기: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]})")
            
            return filtered_boxes
            
        except Exception as e:
            print(f"❌ 초록색 박스 검출 오류: {e}")
            return []
    
    def remove_overlapping_boxes(self, boxes):
        """겹치는 박스들 제거"""
        if len(boxes) <= 1:
            return boxes
        
        boxes.sort(key=lambda x: x['area'], reverse=True)
        
        filtered = []
        for box1 in boxes:
            is_duplicate = False
            bbox1 = box1['bbox']
            
            for box2 in filtered:
                bbox2 = box2['bbox']
                iou = self.calculate_iou(bbox1, bbox2)
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(box1)
        
        return filtered
    
    def calculate_iou(self, box1, box2):
        """두 박스의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def select_best_face_box(self, green_boxes):
        """여러 박스 중 가장 큰 박스 선택 (크기 우선)"""
        if not green_boxes:
            return None
        
        if len(green_boxes) == 1:
            return green_boxes[0]
        
        # 면적 기준으로 정렬 (가장 큰 것부터)
        green_boxes.sort(key=lambda x: x['area'], reverse=True)
        best_box = green_boxes[0]
        
        if self.debug_mode:
            bbox = best_box['bbox']
            print(f"✅ 가장 큰 박스 선택: 면적 {best_box['area']} (크기: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]})")
            if len(green_boxes) > 1:
                print(f"   전체 {len(green_boxes)}개 박스 중에서 선택")
        
        return best_box

    def predict_emotion(self, face_image):
        """감정 분류 추론 (direct_resize_imagenet 방식만 사용)"""
        try:
            if self.debug_mode:
                print(f"\n🧠 감정 분류 시작 - 입력 이미지 크기: {face_image.shape}")
            
            # direct_resize_imagenet 방식만 사용
            face_320 = cv2.resize(face_image, (320, 320))
            face_tensor = self.preprocess_face_imagenet(face_320)
            
            if face_tensor is not None:
                with torch.no_grad():
                    outputs = self.emotion_model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence = torch.max(probabilities).item()
                    predicted = torch.argmax(probabilities, dim=1).item()
                    
                    emotion_classes = ['negative', 'non-negative']
                    final_emotion = emotion_classes[predicted]
                    prob_list = probabilities[0].cpu().numpy().tolist()
                    
                    if self.debug_mode:
                        print(f"📊 감정 분류 결과 (direct_resize_imagenet):")
                        print(f"   {final_emotion} ({confidence:.3f})")
                        print(f"   확률분포: [neg:{prob_list[0]:.3f}, pos:{prob_list[1]:.3f}]")
                        
                        certainty = abs(confidence - 0.5)
                        if certainty > 0.1:
                            print(f"   ⭐ 높은 확신도! (경계선에서 {certainty:.3f} 떨어짐)")
                        else:
                            print(f"   ⚠️ 낮은 확신도 (경계선에서 {certainty:.3f} 떨어짐)")
                    
                    return {
                        'emotion': final_emotion,
                        'confidence': confidence,
                        'probabilities': prob_list,
                        'method_used': 'direct_resize_imagenet',
                        'certainty': abs(confidence - 0.5)
                    }
            else:
                return {'emotion': 'unknown', 'confidence': 0.5, 'probabilities': [0.5, 0.5]}
                
        except Exception as e:
            print(f"감정 분류 오류: {e}")
            return None

    def resize_to_320_method1(self, face_image):
        """320x320 리사이즈 방법 1: 정사각형 패딩 후 리사이즈"""
        h, w = face_image.shape[:2]
        if h != w:
            size = max(h, w)
            square_crop = np.zeros((size, size, 3), dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square_crop[y_offset:y_offset+h, x_offset:x_offset+w] = face_image
            face_square = square_crop
        else:
            face_square = face_image
        
        face_320 = cv2.resize(face_square, (320, 320), interpolation=cv2.INTER_LINEAR)
        return face_320
    
    def preprocess_face_imagenet(self, face_image_320):
        """전처리: ImageNet 정규화 (320x320 입력)"""
        try:
            if face_image_320.shape[:2] != (320, 320):
                face_image_320 = cv2.resize(face_image_320, (320, 320))
            
            face_rgb = cv2.cvtColor(face_image_320, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            face_normalized = (face_normalized - mean) / std
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            face_tensor = torch.from_numpy(face_transposed).unsqueeze(0).float().to(self.device)
            return face_tensor
        except Exception as e:
            print(f"ImageNet 전처리 오류: {e}")
            return None
    
    def preprocess_face_simple(self, face_image_320):
        """전처리: 단순 0~1 정규화 (320x320 입력)"""
        try:
            if face_image_320.shape[:2] != (320, 320):
                face_image_320 = cv2.resize(face_image_320, (320, 320))
            
            face_rgb = cv2.cvtColor(face_image_320, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            face_tensor = torch.from_numpy(face_transposed).unsqueeze(0).float().to(self.device)
            return face_tensor
        except Exception as e:
            print(f"단순 전처리 오류: {e}")
            return None

    def load_emotion_model(self):
        """감정분류 모델 로드"""
        try:
            print(f"감정모델 로딩 중: {self.emotion_model_path}")
            model_data = torch.load(self.emotion_model_path, map_location=self.device)
            if isinstance(model_data, dict):
                try:
                    import timm
                    self.emotion_model = timm.create_model('efficientnet_b0', num_classes=2, pretrained=False)
                    self.emotion_model.load_state_dict(model_data, strict=False)
                except ImportError:
                    print("timm 라이브러리가 설치되지 않음. pip install timm 으로 설치하세요.")
                    return False
            else:
                self.emotion_model = model_data
            self.emotion_model.to(self.device)
            self.emotion_model.eval()
            print("감정모델 로드 완료")
            return True
        except Exception as e:
            print(f"감정모델 로드 실패: {e}")
            return False
    
    def setup_serial(self):
        """시리얼 통신 설정"""
        try:
            print(f"시리얼 포트 연결 시도: {self.serial_port} @ {self.serial_baudrate} baud")
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                timeout=1
            )
            print(f"시리얼 연결 성공: {self.serial_port}")
            return True
        except Exception as e:
            print(f"시리얼 연결 실패: {e}")
            return False
    
    def setup_rtsp(self):
        """RTSP 스트림 설정"""
        try:
            print(f"RTSP 연결 시도: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                print(f"RTSP 연결 실패: {self.rtsp_url}")
                return False
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = width
            self.frame_height = height
            print(f"RTSP 연결 성공: {width}x{height} @ {fps}fps")
            return True
        except Exception as e:
            print(f"RTSP 설정 실패: {e}")
            return False
    
    def parse_yolo_detection(self, data_line):
        """CatchCAM YOLO detection 데이터 파싱 (트리거 용도, 로그 숨김)"""
        try:
            if "[AI coordinate]" in data_line and "Count" in data_line:
                # 로그 출력 제거 (편의성을 위해)
                parts = data_line.split()
                count_idx = parts.index("Count") + 2
                count = int(parts[count_idx])
                if count > 0:
                    coords = {}
                    for i, part in enumerate(parts):
                        if part in ['score']:
                            if i + 2 < len(parts) and parts[i+1] == '=':
                                coords[part] = float(parts[i+2])
                    return {
                        'count': count,
                        'score': coords.get('score', 0.0),
                        'timestamp': time.time()
                    }
        except Exception as e:
            print(f"YOLO 데이터 파싱 오류: {e}")
        return None

    def process_green_box_detection(self, frame, detection_score):
        """초록색 박스 검출 및 감정분석 (1.5배 확장 크롭)"""
        try:
            green_boxes = self.detect_green_boxes(frame)
            
            if not green_boxes:
                print("❌ 초록색 박스를 찾을 수 없음")
                return None, None, None, None, None
            
            best_box = self.select_best_face_box(green_boxes)
            
            if not best_box:
                print("❌ 적절한 얼굴 박스를 찾을 수 없음")
                return None, None, None, None, None
            
            bbox = best_box['bbox']
            x1, y1, x2, y2 = bbox
            
            # 원본 박스 크기 계산
            original_width = x2 - x1
            original_height = y2 - y1
            
            # 1.5배 확장을 위한 새 크기 계산
            expand_factor = 1.5
            new_width = int(original_width * expand_factor)
            new_height = int(original_height * expand_factor)
            
            # 중앙점 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 1.5배 확장된 새 좌표 계산
            expanded_x1 = center_x - new_width // 2
            expanded_y1 = center_y - new_height // 2
            expanded_x2 = center_x + new_width // 2
            expanded_y2 = center_y + new_height // 2
            
            # 프레임 경계 내로 제한
            expanded_x1 = max(0, expanded_x1)
            expanded_y1 = max(0, expanded_y1)
            expanded_x2 = min(self.frame_width, expanded_x2)
            expanded_y2 = min(self.frame_height, expanded_y2)
            
            # 확장된 영역으로 크롭
            face_crop = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2].copy()
            if face_crop.size == 0:
                print("❌ 확장된 크롭 영역이 비어있음!")
                return None, None, None, None, None

            if self.debug_mode:
                print(f"🖼️  원본 박스: {original_width}x{original_height}")
                print(f"🖼️  확장된 크롭: {face_crop.shape[1]}x{face_crop.shape[0]} (1.5배 확장)")
                print(f"   원본 좌표: [{x1}, {y1}] -> [{x2}, {y2}]")
                print(f"   확장 좌표: [{expanded_x1}, {expanded_y1}] -> [{expanded_x2}, {expanded_y2}]")

            emotion_result = self.predict_emotion(face_crop)

            result_frame = frame.copy()
            # 원본 초록 박스 표시 (참고용)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 확장된 크롭 영역 표시 (빨간 테두리)
            cv2.rectangle(result_frame, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (0, 0, 255), 3)
            
            if emotion_result:
                certainty = emotion_result.get('certainty', 0)
                method = emotion_result.get('method_used', 'unknown')
                text = f"CROP: {emotion_result['emotion']} ({emotion_result['confidence']:.2f}) [1.5x]"
                if certainty > 0.1:
                    text += " ⭐"
            else:
                text = f"CROP (Score: {detection_score:.2f}) [1.5x]"
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(0, expanded_y1 - 10)
            cv2.rectangle(result_frame, (expanded_x1, y_text-20), (expanded_x1+text_width+10, y_text+5), (0, 0, 255), -1)
            cv2.putText(result_frame, text, (expanded_x1+5, y_text-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 확장된 좌표를 bbox로 반환 (저장용)
            expanded_bbox = [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
            
            return result_frame, face_crop, emotion_result, expanded_bbox, best_box
        except Exception as e:
            print(f"❌ 초록 박스 처리 오류: {e}")
            return None, None, None, None, None

    def save_result(self, frame, face_crop, bbox, emotion_result, result_frame, detection_score, box_info):
        """결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            result_data = {
                'timestamp': timestamp,
                'detection_score': float(detection_score),
                'bbox': bbox,
                'emotion': emotion_result if emotion_result else {
                    'emotion': 'unknown', 'confidence': 0.0, 'probabilities': []
                },
                'green_box_detection': {
                    'method': 'color_based_detection',
                    'box_area': box_info['area'],
                    'aspect_ratio': box_info['aspect_ratio'],
                    'rectangularity': box_info['rectangularity']
                },
                'image_files': {
                    'original': f"{timestamp}_original.jpg",
                    'face_crop': f"{timestamp}_face.jpg"
                },
                'frame_size': {'width': frame.shape[1], 'height': frame.shape[0]},
                'face_crop_size': {'width': face_crop.shape[1], 'height': face_crop.shape[0]}
            }
            
            if result_frame is not None and self.save_visualization:
                result_data['image_files']['result'] = f"{timestamp}_result.jpg"
                cv2.imwrite(str(self.output_dir / result_data['image_files']['result']), result_frame)
            
            cv2.imwrite(str(self.output_dir / result_data['image_files']['original']), frame)
            cv2.imwrite(str(self.output_dir / result_data['image_files']['face_crop']), face_crop)
            
            with open(self.output_dir / f"{timestamp}_result.json", 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 저장 완료: {timestamp}")
            if emotion_result:
                print(f"   감정: {emotion_result['emotion']} ({emotion_result['confidence']:.2f})")
            print(f"   YOLO 신뢰도: {detection_score:.3f}")
            print(f"   검출된 박스: [{bbox[0]}, {bbox[1]}] -> [{bbox[2]}, {bbox[3]}]")
            print(f"   크롭 크기: {face_crop.shape[1]}x{face_crop.shape[0]}  [green_box_detection]")
            return True
        except Exception as e:
            print(f"결과 저장 오류: {e}")
            return False

    def serial_reader_thread(self):
        """시리얼 데이터 읽기 스레드 (2초 간격 + 2초 안정화 시간)"""
        print("시리얼 리더 스레드 시작 - YOLO 검출 트리거 대기")
        last_detection_time = 0
        min_detection_interval = 2.0  # 2초로 변경
        first_detection_time = None  # 첫 검출 시간 기록
        stabilization_period = 2.0   # 2초 안정화 시간
        
        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        detection_data = self.parse_yolo_detection(line)
                        if detection_data:
                            current_time = time.time()
                            
                            # 첫 검출인지 확인
                            if first_detection_time is None:
                                first_detection_time = current_time
                                print(f"\n🎯 첫 YOLO 검출! 안정화를 위해 {stabilization_period}초 대기...")
                                continue
                            
                            # 안정화 시간이 지났는지 확인
                            if current_time - first_detection_time < stabilization_period:
                                remaining_time = stabilization_period - (current_time - first_detection_time)
                                print(f"⏳ 안정화 대기 중... (남은 시간: {remaining_time:.1f}초)")
                                continue
                            
                            # 최소 검출 간격 확인
                            if current_time - last_detection_time < min_detection_interval:
                                continue
                            
                            print(f"\n{'='*60}")
                            print(f"🎯 YOLO 트리거! 초록 박스 검출 시작 (250x250 최소 크기)")
                            print(f"   신뢰도: {detection_data['score']:.3f}")
                            
                            current_frame = None
                            with self.frame_lock:
                                if self.latest_frame is not None:
                                    current_frame = self.latest_frame.copy()
                            
                            if current_frame is not None:
                                result = self.process_green_box_detection(
                                    current_frame, detection_data['score']
                                )
                                
                                if result and len(result) == 5:
                                    result_frame, face_crop, emotion_result, bbox, box_info = result
                                    if face_crop is not None:
                                        saved = self.save_result(
                                            current_frame, face_crop, bbox, emotion_result,
                                            result_frame, detection_data['score'], box_info
                                        )
                                        if saved:
                                            last_detection_time = current_time
                                            print(f"{'='*60}\n")
                            else:
                                print("⏳ 프레임이 아직 준비되지 않음")
                        elif "[SEND]" in line or "[RECV]" in line:
                            if "inf_number" not in line:
                                print(f"Serial: {line}")
            except Exception as e:
                print(f"시리얼 읽기 오류: {e}")
                time.sleep(0.1)
    
    def frame_capture_thread(self):
        """RTSP 프레임 캡처 스레드"""
        print("프레임 캡처 스레드 시작")
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frame = frame
                    frame_count += 1
                    error_count = 0
                    if frame_count % 300 == 0:
                        print(f"📹 프레임 캡처 중... (총 {frame_count} 프레임)")
                else:
                    error_count += 1
                    if error_count >= max_errors:
                        print(f"⚠️  프레임 읽기 실패 {error_count}회 - 재연결 시도")
                        self.reconnect_rtsp()
                        error_count = 0
                    time.sleep(0.1)
            except Exception as e:
                print(f"프레임 캡처 오류: {e}")
                time.sleep(0.1)
    
    def reconnect_rtsp(self):
        """RTSP 재연결"""
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            time.sleep(1)
            self.setup_rtsp()
        except Exception as e:
            print(f"RTSP 재연결 실패: {e}")
    
    def run(self):
        """메인 실행 루프"""
        print("\n" + "="*60)
        print("🚪 Door-Box 인퍼런스 시스템 (초록 박스 검출)")
        print("   YOLO 트리거 → 초록 박스 자동 검출 → 감정 분류")
        print("   좌표 변환 없이 그려진 박스를 직접 인식!")
        print("="*60 + "\n")
        
        if not self.load_emotion_model():
            return False
        if not self.setup_serial():
            return False
        if not self.setup_rtsp():
            return False
        
        self.running = True
        
        serial_thread = threading.Thread(target=self.serial_reader_thread, daemon=True)
        frame_thread = threading.Thread(target=self.frame_capture_thread, daemon=True)
        serial_thread.start()
        frame_thread.start()
        
        print("\n✅ 모든 스레드 시작 완료")
        print("🔍 YOLO 트리거 대기 중... (종료: Ctrl+C)\n")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n종료 신호 받음...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("리소스 정리 중...")
        self.running = False
        if hasattr(self, 'serial_conn'):
            try:
                self.serial_conn.close()
            except Exception:
                pass
        if hasattr(self, 'cap'):
            try:
                self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("✅ 정리 완료\n")

def main():
    parser = argparse.ArgumentParser(description='Door-Box 초록 박스 검출 시스템')
    parser.add_argument('--rtsp_url', default='rtsp://10.0.0.156/live1.sdp', help='CatchCAM RTSP URL')
    parser.add_argument('--serial_port', default='/dev/ttyUSB0', help='시리얼 포트')
    parser.add_argument('--serial_baudrate', type=int, default=115200, help='시리얼 통신 속도')
    parser.add_argument('--emotion_model', required=True, help='감정분류 모델 경로 (.pth 파일)')
    parser.add_argument('--output_dir', default='output', help='결과 저장 디렉토리')
    parser.add_argument('--no-visualization', action='store_true', help='시각화 이미지 저장 안함')
    args = parser.parse_args()
    
    doorbox = DoorBoxInference(
        rtsp_url=args.rtsp_url,
        serial_port=args.serial_port,
        serial_baudrate=args.serial_baudrate,
        emotion_model_path=args.emotion_model,
        output_dir=args.output_dir,
        save_visualization=not args.no_visualization
    )
    doorbox.run()

if __name__ == "__main__":
    main()
