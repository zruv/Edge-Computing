#!/bin/bash

echo "Starting Termux AI Setup (Final Dependency Fix)..."

# 1. Install the missing library
echo "Installing dbus..."
pkg install dbus -y

# 2. Verify everything one last time
echo "---------------------------------------------------"
if python -c "import cv2; print('SUCCESS: OpenCV is working!')"; then
    echo "---------------------------------------------------"
    echo ""
    echo "GREAT SUCCESS! The AI is ready."
    echo "Run: python detect.py"
else
    echo "---------------------------------------------------"
    echo "Still missing something. Check the error above."
fi