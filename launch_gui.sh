#!/bin/bash
# PCC-Explorer Launch Script for Mac/Linux

echo "Activating Python virtual environment..."

# Check if the venv exists before trying to activate it
if [ ! -f "./venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run the installation steps in README.md first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source ./venv/bin/activate

echo "Launching PCC-Explorer GUI..."
python -m src.main_gui

echo "GUI closed."
