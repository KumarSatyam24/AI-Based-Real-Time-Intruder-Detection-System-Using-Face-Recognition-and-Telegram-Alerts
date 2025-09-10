# Quick Start Guide

Get your Facial Detection System running in under 10 minutes!

## 🚀 One-Command Setup

```bash
# Clone and setup in one go
git clone <repository-url> && cd facial_detection_system && ./scripts/start_system.sh setup
```

## 📱 Telegram Setup (2 minutes)

1. **Create Bot** - Message [@BotFather](https://t.me/botfather):
   ```
   /newbot
   My Security Bot
   my_security_bot
   ```
   → Save the token! 🔑

2. **Get Chat ID**:
   - Message your bot: `Hello`
   - Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Copy the chat ID number

3. **Configure**:
   ```bash
   nano config/.env
   ```
   ```env
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   TELEGRAM_CHAT_ID=123456789
   ```

## 👥 Add Authorized People (1 minute)

```bash
./scripts/start_system.sh database
# Choose option 3 - Add from camera
# Enter name, take 5 photos, done!
```

## 🎯 Start Monitoring

```bash
./scripts/start_system.sh start
```

**You're live!** 📹✨

## ✅ Test Everything

```bash
# Test alerts
python src/main.py --test-telegram

# Check camera
./scripts/start_system.sh status
```

## 🆘 Quick Fixes

| Problem | Solution |
|---------|----------|
| No camera | Try `camera: source: 1` in config |
| No alerts | Check bot token in `.env` |
| Slow performance | Set `frame_skip_rate: 3` |
| False alarms | Lower `threshold` to `0.4` |

## 📞 Need Help?

1. **Logs**: `tail -f logs/facial_detection.log`
2. **Status**: `./scripts/start_system.sh status`  
3. **Restart**: `./scripts/start_system.sh restart`

---

**That's it! Your security system is now protecting you 24/7** 🛡️

*Full documentation: [README.md](README.md)*
