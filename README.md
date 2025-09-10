# Facial Detection System

A comprehensive real-time facial detection and recognition system with Telegram alerts for security monitoring.

## 🚀 Features

- **Real-time Face Detection**: Advanced face detection using OpenCV Haar Cascades and DNN models
- **Face Recognition**: Identify authorized personnel vs. unknown intruders
- **Telegram Alerts**: Instant notifications with intruder photos sent to your phone
- **Database Management**: Easy-to-use interface for managing authorized personnel
- **Performance Optimization**: Efficient processing for real-time monitoring
- **Comprehensive Logging**: Detailed system logs for monitoring and debugging
- **Configurable Settings**: Flexible configuration for different deployment scenarios

## 📋 Requirements

### Hardware Requirements
- **Camera**: USB webcam or IP camera
- **CPU**: Multi-core processor (2+ cores recommended)
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space

### Software Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **Internet Connection**: Required for Telegram alerts

## 🛠 Installation

### Quick Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd facial_detection_system
   ```

2. **Run the automated setup**:
   ```bash
   ./scripts/start_system.sh setup
   ```

3. **Configure your settings**:
   ```bash
   # Edit the configuration file
   nano config/.env
   ```
   
   Add your Telegram bot credentials:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

### Manual Setup

1. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create directories**:
   ```bash
   mkdir -p data/authorized_faces data/captured_intruders logs models
   ```

## 🔧 Configuration

### Telegram Bot Setup

1. **Create a Telegram Bot**:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Send `/newbot` and follow instructions
   - Save the bot token

2. **Get your Chat ID**:
   - Message your bot once
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat ID in the response

3. **Configure the system**:
   ```bash
   cp config/.env.example config/.env
   nano config/.env
   ```

### Camera Configuration

Edit `config/settings.yaml`:

```yaml
camera:
  source: 0  # 0 for default webcam, or "http://ip:port/stream" for IP camera
  frame_width: 640
  frame_height: 480
  frame_skip_rate: 2  # Process every nth frame
```

### Recognition Settings

```yaml
face_recognition:
  threshold: 0.6  # Lower = more strict, Higher = more lenient
  tolerance: 0.6  # Face matching tolerance
  model: "hog"    # "hog" for speed, "cnn" for accuracy
```

## 🎯 Usage

### Starting the System

```bash
# Start the system
./scripts/start_system.sh start

# Check status
./scripts/start_system.sh status

# Stop the system
./scripts/start_system.sh stop
```

### Managing Authorized Personnel

```bash
# Open database manager
./scripts/start_system.sh database
```

Or directly:
```bash
python src/main.py --database-manager
```

#### Adding People

1. **From Images**:
   - Select option 2 in database manager
   - Enter person's name and department
   - Provide paths to their photos

2. **From Camera**:
   - Select option 3 in database manager
   - Enter person's name
   - Follow on-screen instructions to capture photos

### Testing the System

```bash
# Run all tests
./scripts/start_system.sh test

# Test Telegram connection only
python src/main.py --test-telegram
```

## 📱 Mobile App Integration

### Telegram Commands

Once your system is running, you can interact with it via Telegram:

- **System Status**: Automatic startup/shutdown notifications
- **Intruder Alerts**: Real-time alerts with photos
- **System Health**: Error notifications and status updates

### Alert Types

1. **Intruder Detection**: 
   - Photo of unknown person
   - Timestamp and confidence level
   - Location information

2. **System Status**:
   - Startup confirmation
   - Error notifications
   - Shutdown alerts

## 🔍 Monitoring & Logs

### Log Files

- **Application Logs**: `logs/facial_detection.log`
- **Error Logs**: Check console output or log file
- **System Performance**: Built-in FPS and performance metrics

### Performance Monitoring

The system displays real-time metrics:
- **FPS**: Frames processed per second
- **Detection Count**: Number of faces detected
- **Recognition Count**: Number of recognition attempts
- **Alert Count**: Number of alerts sent

## 🛡️ Security Considerations

### Privacy & Data Protection

- **Local Storage**: All face data stored locally
- **Encrypted Communications**: Telegram uses end-to-end encryption
- **Access Control**: Limit physical access to the system
- **Data Retention**: Configure automatic cleanup of old intruder images

### Best Practices

1. **Regular Updates**: Keep dependencies updated
2. **Secure Network**: Use secure network for IP cameras
3. **Access Logs**: Monitor system access logs
4. **Backup**: Regular backup of authorized personnel database

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera access
ls /dev/video*  # Linux
# or
system_profiler SPCameraDataType  # macOS
```

**Solutions**:
- Verify camera is connected and not used by other applications
- Try different camera source numbers (0, 1, 2...)
- Check camera permissions in system settings

#### Face Recognition Not Working
**Symptoms**: All faces detected as "Unknown"

**Solutions**:
1. Add more photos of authorized personnel (5-10 recommended)
2. Ensure good lighting in reference photos
3. Adjust recognition threshold in settings
4. Verify face encodings database is not corrupted

#### Telegram Alerts Not Working
**Symptoms**: No alerts received on phone

**Solutions**:
1. Test connection: `python src/main.py --test-telegram`
2. Verify bot token and chat ID in `.env` file
3. Check internet connection
4. Ensure bot is not blocked or limited

#### Poor Performance
**Symptoms**: Low FPS, system lag

**Solutions**:
1. Reduce camera resolution in config
2. Increase frame skip rate
3. Use "hog" model instead of "cnn"
4. Close other applications using camera/CPU

### Debug Mode

Enable debug logging by editing `config/settings.yaml`:

```yaml
logging:
  level: "DEBUG"
  console_output: true
```

### Getting Help

1. **Check Logs**: Review `logs/facial_detection.log`
2. **Test Individual Components**: Use the test suite
3. **Verify Configuration**: Double-check all settings
4. **System Resources**: Ensure adequate CPU/memory

## 🔄 System Maintenance

### Regular Tasks

1. **Weekly**:
   - Review intruder images and alerts
   - Check system performance metrics
   - Update authorized personnel as needed

2. **Monthly**:
   - Clean up old intruder images
   - Review and rotate logs
   - Update system dependencies

3. **As Needed**:
   - Add/remove authorized personnel
   - Adjust recognition thresholds
   - Update Telegram settings

### Database Maintenance

```bash
# Export database info
./scripts/start_system.sh database
# Select option 7 to export

# Clean up old images (automatically done)
# Images older than 30 days are removed by default
```

### Performance Optimization

1. **Hardware Optimization**:
   - Use SSD for faster I/O
   - Ensure adequate cooling
   - Consider dedicated GPU for CNN models

2. **Software Optimization**:
   - Adjust frame skip rate
   - Optimize camera resolution
   - Use appropriate face detection method

## 📊 System Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Face Detection  │───▶│ Face Recognition│
│   (Camera)      │    │   (OpenCV)       │    │ (face_recognition)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│ Telegram Alerts │◀───│   Main System    │◀────────────┘
│    (Intruders)  │    │   Controller     │
└─────────────────┘    └──────────────────┘
                                │
                       ┌──────────────────┐
                       │   Face Database  │
                       │  (Authorized)    │
                       └──────────────────┘
```

### Data Flow

1. **Video Capture** → Captures frames from camera
2. **Face Detection** → Detects faces in frames
3. **Face Recognition** → Identifies faces against database
4. **Alert System** → Sends notifications for unknown faces
5. **Database Management** → Manages authorized personnel

## 🚀 Advanced Features

### Custom Integration

The system is designed to be extensible:

- **API Endpoints**: Add REST API for external integration
- **Multiple Cameras**: Extend to support multiple camera sources
- **Cloud Storage**: Integrate with cloud storage for backups
- **Mobile App**: Develop dedicated mobile application

### Performance Scaling

For high-performance deployments:

- **GPU Acceleration**: Use CUDA-enabled OpenCV
- **Distributed Processing**: Split detection and recognition
- **Load Balancing**: Multiple system instances
- **Database Scaling**: Use external database systems

## 📝 License & Support

### License
This project is provided as-is for educational and security purposes.

### Support
For technical support:
1. Check this documentation
2. Review system logs
3. Test individual components
4. Verify hardware compatibility

### Contributing
Contributions are welcome! Please:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Test thoroughly before submitting

---

## 📞 Emergency Procedures

### System Failure
If the system stops working:
1. Check system status: `./scripts/start_system.sh status`
2. Review recent logs: `tail -f logs/facial_detection.log`
3. Restart system: `./scripts/start_system.sh restart`
4. If persistent, check hardware connections

### Security Incident
If unauthorized access is detected:
1. Review captured intruder images
2. Check system logs for unusual activity  
3. Verify authorized personnel database
4. Consider updating security protocols

### False Alerts
To reduce false positive alerts:
1. Add more reference photos of authorized personnel
2. Adjust recognition threshold (increase for less sensitive)
3. Improve lighting conditions
4. Clean camera lens

---

*Last Updated: December 2024*
*Version: 1.0.0*
