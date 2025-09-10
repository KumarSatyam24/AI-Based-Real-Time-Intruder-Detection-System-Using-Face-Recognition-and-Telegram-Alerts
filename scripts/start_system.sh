#!/bin/bash

# Facial Detection System Startup Script
# Epic 9: Deployment & Documentation - Story Point 28

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_DIR/.venv"
MAIN_SCRIPT="$PROJECT_DIR/src/main.py"
CONFIG_FILE="$PROJECT_DIR/config/settings.yaml"
LOCK_FILE="/tmp/facial_detection_system.lock"
LOG_DIR="$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if system is already running
check_if_running() {
    if [ -f "$LOCK_FILE" ]; then
        PID=$(cat "$LOCK_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            log_error "Facial Detection System is already running (PID: $PID)"
            exit 1
        else
            log_warning "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
}

# Function to create lock file
create_lock() {
    echo $$ > "$LOCK_FILE"
}

# Function to remove lock file
remove_lock() {
    rm -f "$LOCK_FILE"
}

# Function to setup signal handlers
setup_signals() {
    trap 'log_info "Received interrupt signal. Shutting down..."; remove_lock; exit 0' INT TERM
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Python virtual environment exists
    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment not found at $VENV_PATH"
        log_info "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    
    # Check if main script exists
    if [ ! -f "$MAIN_SCRIPT" ]; then
        log_error "Main script not found at $MAIN_SCRIPT"
        exit 1
    fi
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found at $CONFIG_FILE"
        exit 1
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_DIR/config/.env" ]; then
        log_warning ".env file not found. Creating from template..."
        if [ -f "$PROJECT_DIR/config/.env.example" ]; then
            cp "$PROJECT_DIR/config/.env.example" "$PROJECT_DIR/config/.env"
            log_warning "Please edit config/.env with your Telegram bot credentials"
        fi
    fi
    
    log_info "Prerequisites check completed"
}

# Function to activate virtual environment
activate_venv() {
    log_info "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    
    # Verify Python path
    PYTHON_PATH=$(which python)
    if [[ "$PYTHON_PATH" != *".venv"* ]]; then
        log_error "Failed to activate virtual environment"
        exit 1
    fi
    
    log_info "Virtual environment activated: $PYTHON_PATH"
}

# Function to check camera availability
check_camera() {
    log_info "Checking camera availability..."
    
    python << EOF
import cv2
import sys

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera")
        sys.exit(1)
    else:
        print("Camera check passed")
    cap.release()
except Exception as e:
    print(f"ERROR: Camera check failed: {e}")
    sys.exit(1)
EOF

    if [ $? -ne 0 ]; then
        log_error "Camera check failed"
        exit 1
    fi
}

# Function to test Telegram connection
test_telegram() {
    log_info "Testing Telegram connection..."
    
    cd "$PROJECT_DIR"
    python "$MAIN_SCRIPT" --test-telegram
    
    if [ $? -eq 0 ]; then
        log_info "Telegram connection test passed"
    else
        log_warning "Telegram connection test failed. System will start without alerts."
    fi
}

# Function to start the system
start_system() {
    log_info "Starting Facial Detection System..."
    
    cd "$PROJECT_DIR"
    
    # Create lock file
    create_lock
    
    # Start the main application
    python "$MAIN_SCRIPT" --config "$CONFIG_FILE"
}

# Function to show system status
show_status() {
    if [ -f "$LOCK_FILE" ]; then
        PID=$(cat "$LOCK_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            log_info "Facial Detection System is running (PID: $PID)"
            
            # Show additional info if available
            if command -v ps >/dev/null 2>&1; then
                echo -e "${BLUE}Process Info:${NC}"
                ps -p $PID -o pid,ppid,user,cpu,mem,etime,cmd
            fi
        else
            log_warning "Lock file exists but process is not running"
            rm -f "$LOCK_FILE"
        fi
    else
        log_info "Facial Detection System is not running"
    fi
}

# Function to stop the system
stop_system() {
    if [ -f "$LOCK_FILE" ]; then
        PID=$(cat "$LOCK_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            log_info "Stopping Facial Detection System (PID: $PID)..."
            kill -TERM $PID
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    log_info "System stopped gracefully"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if ps -p $PID > /dev/null 2>&1; then
                log_warning "Force killing process..."
                kill -KILL $PID
            fi
            
            remove_lock
        else
            log_warning "Lock file exists but process is not running"
            remove_lock
        fi
    else
        log_info "Facial Detection System is not running"
    fi
}

# Function to restart the system
restart_system() {
    stop_system
    sleep 2
    start_system
}

# Function to show help
show_help() {
    echo -e "${BLUE}Facial Detection System Control Script${NC}"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start the facial detection system"
    echo "  stop      Stop the facial detection system"
    echo "  restart   Restart the facial detection system"
    echo "  status    Show system status"
    echo "  test      Run system tests"
    echo "  setup     Setup the system for first run"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start the system"
    echo "  $0 stop           # Stop the system"
    echo "  $0 restart        # Restart the system"
    echo "  $0 status         # Check if system is running"
    echo ""
}

# Function to run system tests
run_tests() {
    log_info "Running system tests..."
    
    activate_venv
    
    cd "$PROJECT_DIR"
    
    # Check if pytest is available
    if ! python -c "import pytest" 2>/dev/null; then
        log_error "pytest not available. Please install test dependencies:"
        log_info "pip install pytest pytest-asyncio"
        exit 1
    fi
    
    # Run tests
    python -m pytest tests/ -v
    
    if [ $? -eq 0 ]; then
        log_info "All tests passed"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Function to setup system for first run
setup_system() {
    log_info "Setting up Facial Detection System..."
    
    # Check Python version
    python3 --version || {
        log_error "Python 3 is required but not installed"
        exit 1
    }
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_PATH" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    fi
    
    # Activate virtual environment
    activate_venv
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
    
    # Create necessary directories
    log_info "Creating directories..."
    mkdir -p "$PROJECT_DIR/data/authorized_faces"
    mkdir -p "$PROJECT_DIR/data/captured_intruders"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/models"
    
    # Copy environment file if it doesn't exist
    if [ ! -f "$PROJECT_DIR/config/.env" ]; then
        if [ -f "$PROJECT_DIR/config/.env.example" ]; then
            cp "$PROJECT_DIR/config/.env.example" "$PROJECT_DIR/config/.env"
            log_info "Created .env file from template"
            log_warning "Please edit config/.env with your configuration"
        fi
    fi
    
    # Run basic tests
    log_info "Running basic system check..."
    python -c "import cv2, face_recognition, yaml; print('Basic imports successful')"
    
    log_info "System setup completed successfully!"
    log_info "Next steps:"
    echo -e "${YELLOW}  1. Edit config/.env with your Telegram bot credentials${NC}"
    echo -e "${YELLOW}  2. Add authorized faces using: $0 database${NC}"
    echo -e "${YELLOW}  3. Start the system using: $0 start${NC}"
}

# Function to open database manager
open_database_manager() {
    log_info "Opening Database Manager..."
    
    activate_venv
    cd "$PROJECT_DIR"
    
    python "$MAIN_SCRIPT" --database-manager
}

# Main script logic
main() {
    case "${1:-}" in
        "start")
            check_if_running
            setup_signals
            check_prerequisites
            activate_venv
            check_camera
            test_telegram
            start_system
            ;;
        "stop")
            stop_system
            ;;
        "restart")
            restart_system
            ;;
        "status")
            show_status
            ;;
        "test")
            run_tests
            ;;
        "setup")
            setup_system
            ;;
        "database")
            open_database_manager
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        "")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
