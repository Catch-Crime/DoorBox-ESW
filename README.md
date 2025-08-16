# DoorBox - Real-time Face Analysis System

An embedded AI system for real-time face detection and multi-modal classification using Raspberry Pi 5 and CatchCAM KL630.

## Overview

DoorBox is an intelligent door monitoring system that performs real-time face analysis including emotion detection, mask detection, gender classification, and age estimation. The system integrates with AWS S3 for data storage and provides comprehensive analytics for visitor monitoring.

## System Architecture

### Hardware Components
- **Raspberry Pi 5 (8GB)** - Main processing unit running Ubuntu 24.04 ARM64
- **CatchCAM KL630 96Board** - AI camera with pre-loaded YOLO detection model
- **PIR Motion Sensor** - Trigger for detection activation
- **RGB LED** - Status indication
- **OLED Display** - Progress monitoring (optional)

### Software Stack
- **OS**: Ubuntu 24.04 LTS ARM64
- **Runtime**: Python 3.11 with virtual environment
- **AI Framework**: PyTorch 2.x ARM64
- **Computer Vision**: OpenCV, PIL
- **Cloud Storage**: AWS S3 with boto3
- **Communication**: Serial (UART), RTSP streaming

## Features

### Multi-Modal Classification
- **Emotion Detection**: Binary classification (negative/non-negative)
- **Mask Detection**: Presence/absence with confidence threshold
- **Gender Classification**: Male/female identification
- **Age Estimation**: 9-class age group prediction (0-9, 10-19, ..., 80+)

### Smart Detection Logic
- **Conditional Processing**: Mask detection runs first; if mask detected, other classifications are skipped
- **Confidence Thresholding**: Mask detection requires 0.7+ confidence for positive classification
- **Session Management**: 20-second detection sessions with 3-second capture intervals

### Data Management
- **Local Storage**: Temporary storage with structured directories
- **AWS S3 Integration**: Automated batch upload every 60 seconds
- **File Organization**: Date-based folder structure with timestamped files
- **Data Format**: JSON metadata + JPEG frames + MP4 video clips (5-second buffers)

## Model Architecture

| Component | Model | Purpose | Classes |
|-----------|-------|---------|---------|
| Emotion | EfficientNet-B0 | Sentiment analysis | 2 (negative, non-negative) |
| Accessory | GhostNet | Mask detection | 2 (with/without mask) |
| Gender | MobileNetV3-Small | Gender classification | 2 (male, female) |
| Age | EfficientNet-B0 | Age group estimation | 9 (decade-based groups) |

## Installation

### Prerequisites
```bash
# System dependencies
sudo apt update && sudo apt install python3-venv git

# Python virtual environment
python3 -m venv ~/doorbox
source ~/doorbox/bin/activate

# Required packages
pip install torch torchvision opencv-python-headless
pip install timm boto3 pyserial pytz
```

### Configuration Setup
1. Create `config.py` with your settings:
```python
# AWS Configuration
AWS_ACCESS_KEY_ID = "your_access_key"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
AWS_BUCKET_NAME = "doorbox-data"
AWS_REGION = "ap-northeast-2"

# Device Configuration
DEVICE_ID = "doorbox_001"
RTSP_URL = "rtsp://10.0.0.156/live1.sdp"
SERIAL_PORT = "/dev/ttyUSB0"

# Detection Parameters
DETECTION_DELAY = 0          # Immediate capture
CAPTURE_INTERVAL = 3         # 3-second intervals
DETECTION_TIMEOUT = 20       # 20-second sessions
```

2. Place model files in `models/` directory:
   - `emotion2.pth` - Emotion classification
   - `acc.pt` - Accessory/mask detection
   - `gender.pt` - Gender classification
   - `age.pt` - Age estimation

### AWS S3 Setup
1. Create S3 bucket with appropriate permissions
2. Configure IAM user with S3 access rights
3. Set up bucket policy for device uploads

## Usage

### Running the System
```bash
# Activate environment
source ~/doorbox/bin/activate

# Start the inference system
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

### Detection Workflow
1. **Motion Detection**: PIR sensor triggers system activation
2. **YOLO Processing**: CatchCAM detects faces and sends coordinates
3. **Face Extraction**: Green bounding boxes are detected and expanded 1.5x
4. **Classification**: Multi-modal analysis with conditional logic
5. **Data Storage**: Results saved locally and queued for S3 upload
6. **Session Management**: Automatic session termination after timeout

## Data Structure

### S3 Bucket Organization
```
doorbox-data/
├── 2025/08/16/
│   ├── 20250816_143052_log/
│   │   ├── 20250816_143052_frame.jpg
│   │   ├── 20250816_143052_clip.mp4
│   │   └── 20250816_143052_result.json
```

### JSON Output Format
```json
{
  "day": "20250816",
  "time": "14:30:52",
  "detection_results": {
    "emotion": "non-negative",
    "confidence": 0.851,
    "has_mask": false,
    "gender": "female",
    "age_group": "20-29"
  },
  "image_key": "doorbox-data/2025/08/16/20250816_143052_log/20250816_143052_frame.jpg"
}
```

## Performance Optimization

### Real-time Processing
- **RTSP Buffer**: Minimized to 1 frame for reduced latency
- **Serial Communication**: 0.1s timeout with 1ms polling
- **Frame Processing**: Immediate response to YOLO detections
- **Memory Management**: Automatic cleanup of processed files

### Resource Efficiency
- **Conditional Classification**: Skip unnecessary processing when mask detected
- **Batch Uploads**: Reduce S3 API calls with 60-second intervals
- **Session Timeout**: Automatic cleanup prevents resource leaks
- **Thread Management**: Daemon threads for background processing

## API Integration

The system stores results in S3 with a structured format suitable for:
- **Mobile Applications**: Real-time visitor analytics
- **Dashboard Systems**: Historical data visualization
- **Analytics Platforms**: Behavioral pattern analysis
- **Alert Systems**: Security and access monitoring

### Suggested API Endpoints
- `GET /detections/latest` - Recent detections
- `GET /detections/{date}` - Daily analytics
- `GET /statistics` - Aggregated insights

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Verify model files exist and have correct permissions
2. **RTSP Connection**: Check CatchCAM IP address and network connectivity
3. **Serial Communication**: Ensure correct port and baud rate configuration
4. **S3 Upload Failures**: Verify AWS credentials and bucket permissions

### Debug Mode
Enable detailed logging by modifying the log level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Hardware Specifications

### Minimum Requirements
- **CPU**: ARM64 architecture (Raspberry Pi 4+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 32GB microSD or SSD
- **Network**: Ethernet connection for stability
- **Power**: 5V 3A stable power supply

### Recommended Setup
- **Raspberry Pi 5 8GB** for optimal performance
- **High-speed microSD** (Class 10, A2) or USB 3.0 SSD
- **Active cooling** for sustained performance
- **UPS/Battery backup** for power reliability

## Development

### Project Structure
```
doorbox/
├── config.py              # Configuration settings
├── inference.py           # Main inference system
├── models/                # AI model files
├── output/                # Local temporary storage
├── logs/                  # System logs
└── README.md              # Documentation
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request with detailed description

## License

This project is proprietary software developed for embedded AI applications. Contact the development team for licensing information.

## Support

For technical support or questions:
- Check troubleshooting guide above
- Review system logs in `logs/doorbox.log`
- Verify hardware connections and configurations
- Contact development team for advanced issues

---

**Note**: This system is designed for controlled environments with proper lighting and positioning. Performance may vary based on environmental conditions and hardware configuration.
