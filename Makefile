.PHONY: help install setup train inference evaluate clean

help:
	@echo "Fine-Tune Custom Models - Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup virtual environment and install dependencies"
	@echo "  make train       - Start fine-tuning with default config"
	@echo "  make inference   - Run inference in interactive mode"
	@echo "  make evaluate    - Evaluate model on test set"
	@echo "  make clean       - Clean output files and caches"

install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate"

train:
	python finetune.py --config config.yaml

inference:
	python inference.py --model-path ./output --model-type causal_lm

evaluate:
	python evaluate.py --model-path ./output --model-type causal_lm --test-file examples/sample_data.json

clean:
	rm -rf output/
	rm -rf logs/
	rm -rf wandb/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned output files and caches"
