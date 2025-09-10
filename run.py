#!/usr/bin/env python3
"""
Facial Detection System Entry Point
Epic 9: Deployment & Documentation - Story Point 30

Simple entry point for running the system directly.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from main import main
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
