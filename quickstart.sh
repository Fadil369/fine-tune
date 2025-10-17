#!/bin/bash

# Quick start script for fine-tuning

echo "====================================="
echo "Fine-Tune Quick Start"
echo "====================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Prepare your data:"
echo "   python prepare_data.py --input your_data.csv --output data.json --task instruction"
echo ""
echo "2. Edit config.yaml to match your use case"
echo ""
echo "3. Start training:"
echo "   python finetune.py --config config.yaml"
echo ""
echo "4. Test your model:"
echo "   python inference.py --model-path ./output --model-type causal_lm"
echo ""
echo "For more information, see README.md"
echo "====================================="
