# DoorBox-ESW

> **Embedded Software Implementation for DoorBox Project**

This repository contains the embedded software implementation for DoorBox's edge AI system. Handles PIR motion detection, CatchCAM integration, AI model inference, and AWS cloud connectivity on Raspberry Pi 5.

## Project Overview

The ESW component manages hardware integration, real-time AI processing, and cloud connectivity for the DoorBox smart monitoring system. Uses conditional processing and multi-threading for efficient resource management.

## Repository Structure

```
DoorBox-ESW/
├── inference.py        # Main system controller
├── config.py          # Configuration template  
├── requirements.txt   # Python dependencies
└── models/           # AI model files (.pt)
    ├── emotion_model.pt
    ├── gender_model.pt
    └── age_model.pt
```

---

## Hardware Setup

### Components
- Raspberry Pi 5 (8GB) + CatchCAM KL630
- PIR Motion Sensor (HC-SR501)
- RGB LED + 0.96" OLED Display

### GPIO Connections
```
RGB LED:  R=Pin12, G=Pin16, B=Pin18, GND=Pin9
OLED:     SDA=Pin3, SCL=Pin5, VCC=Pin1, GND=Pin6  
PIR:      VCC=Pin4, OUT=Pin8, GND=Pin9
CatchCAM: USB + UART (115200 baud)
```

### System States

| State | RGB LED | OLED Display |
|-------|---------|--------------|
| STANDBY | Red | "Standby Mode" |
| PIR_DETECTED | White | "Motion Detected" |
| INFERENCE_ACTIVE | Purple (blinking) | "Face Processing..." |

---

## Installation

```bash
# Clone and setup
git clone https://github.com/Catch-Crime/DoorBox-ESW.git
cd DoorBox-ESW
python3 -m venv ~/doorbox
source ~/doorbox/bin/activate
pip install -r requirements.txt

# Configure AWS credentials in config.py
# Run system
python inference.py
```

### Expected Output
```
✅ S3 버킷 연결 확인: doorbox-data
✅ 감정 분류 모델 로드 완료
✅ 악세서리 모델 로드 완료 (Custom GhostNet)
✅ 연령대 모델 로드 완료 (EfficientNet-B0, 9클래스)
✅ 성별 모델 로드 완료 (MobileNetV3-Small)
RTSP 스트림 시작 (최적화됨)
시리얼 연결 성공: /dev/ttyUSB0
DoorBox 시스템 시작됨
```

### JSON Output Format
```json
{
  "day": "20250823",
  "time": "21:40:06", 
  "image_key": "doorbox-data/home-1/cam-1/2025/08/23/20250823_214006_log/20250823_214006_frame.jpg",
  "detection_results": {
    "accessory": true,
    "emotion": "alert",
    "gender": "male",     // 0=male, 1=female
    "age_group": "20s"    // 0s, 10s, 20s, ..., 70s, over80s
  }
}
```

### S3 Bucket Organization
```
doorbox-data (버킷명)
└── home-1/
    └── cam-1/
        └── 2025/
            └── 08/
                ├── 23/
                │   ├── 20250823_214006_log/
                │   │   ├── 20250823_214006_frame.jpg (원본 프레임)
                │   │   ├── 20250823_214006_clip.mp4 (5초 영상 클립)
                │   │   └── 20250823_214006_result.json (분류 결과)
                │   ├── 20250823_220145_log/
                │   └── ...
                ├── 24/
                │   ├── 20250824_093022_log/
                │   └── ...
                └── 25/
                    └── 20250825_105423_log/
                        └── ...
```
---

## AI Models

- **emotion_model.pt**: EfficientNet-B0 (Alert/Non-Alert)
- **gender_model.pt**: MobileNetV3-Small (Male/Female)  
- **age_model.pt**: EfficientNet-B0 (0s~80s+, 9 classes)

All models use 320×320 input with conditional processing based on green box detection.

## Key Features

- **Conditional Processing**: AI inference only when green boxes detected
- **Timing Control**: 2-second delay + 5-second intervals
- **Multi-threading**: Separate threads for capture, inference, upload
- **Local Buffering**: Temporary storage before AWS S3 upload
- **Error Recovery**: Automatic retry on communication failures

---

**Developer**: 백승찬 (Seungchan Baek) - [@kairos1228](https://github.com/kairos1228)  
**Part of**: DoorBox Project - 제23회 임베디드 소프트웨어 경진대회

---
