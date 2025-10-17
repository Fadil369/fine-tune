# Quick Reference Card

## Installation

```bash
# Using pip
pip install -r requirements.txt

# Using quick start
bash quickstart.sh

# Verify setup
python test_setup.py
```

## Data Preparation

```bash
# Generate sample data
python generate_sample_data.py --type all

# Convert CSV to JSON (instruction format)
python prepare_data.py \
  --input data.csv \
  --output data.json \
  --task instruction \
  --instruction-column question \
  --output-column answer

# Convert for classification
python prepare_data.py \
  --input reviews.csv \
  --output classification.json \
  --task classification \
  --text-column review \
  --label-column sentiment

# Split data (80/10/10)
python prepare_data.py --input data.json --task split
```

## Training

```bash
# Basic training
python finetune.py --config config.yaml

# With custom config
python finetune.py --config examples/config_instruction.yaml

# Monitor with TensorBoard
tensorboard --logdir ./logs
```

## Inference

```bash
# Interactive mode
python inference.py --model-path ./output --model-type causal_lm

# Single prompt
python inference.py \
  --model-path ./output \
  --model-type causal_lm \
  --prompt "Your question here"

# Batch processing
python inference.py \
  --model-path ./output \
  --model-type causal_lm \
  --mode batch \
  --input-file prompts.json \
  --output-file results.json

# With LoRA model
python inference.py \
  --model-path ./output \
  --model-type causal_lm \
  --use-peft
```

## Evaluation

```bash
# Evaluate generation model
python evaluate.py \
  --model-path ./output \
  --model-type causal_lm \
  --test-file test.json

# Evaluate classification model
python evaluate.py \
  --model-path ./output \
  --model-type classification \
  --test-file test.json

# Compare models
python evaluate.py \
  --model-path "./model1,./model2" \
  --model-type causal_lm \
  --test-file test.json
```

## Configuration Quick Reference

### Minimal Config
```yaml
model:
  name: "gpt2"
  type: "causal_lm"

training:
  output_dir: "./output"
  num_epochs: 3
  batch_size: 4

data:
  train_file: "train.json"
```

### With LoRA (Memory Efficient)
```yaml
model:
  name: "gpt2-medium"
  type: "causal_lm"

training:
  output_dir: "./output"
  batch_size: 2
  gradient_accumulation_steps: 8

data:
  train_file: "train.json"

lora:
  use_lora: true
  r: 8
  lora_alpha: 32
```

### For Large Models (4-bit)
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  type: "causal_lm"
  load_in_4bit: true

training:
  batch_size: 1
  gradient_accumulation_steps: 16

lora:
  use_lora: true
```

### Fast Training (FP16)
```yaml
training:
  fp16: true
  batch_size: 8
  gradient_accumulation_steps: 2
```

## Common Tasks

### Text Generation Model
```bash
# 1. Prepare data
python generate_sample_data.py --type text_gen

# 2. Edit config.yaml
#    - Set model.type: "causal_lm"
#    - Set data.train_file

# 3. Train
python finetune.py --config config.yaml

# 4. Test
python inference.py --model-path ./output --model-type causal_lm
```

### Classification Model
```bash
# 1. Prepare data
python prepare_data.py \
  --input data.csv \
  --output classification.json \
  --task classification

# 2. Use examples/config_classification.yaml

# 3. Train
python finetune.py --config examples/config_classification.yaml

# 4. Test
python inference.py --model-path ./output/classification-model --model-type classification
```

### Instruction-Following Model
```bash
# 1. Prepare data
python prepare_data.py \
  --input qa_data.csv \
  --output instruction.json \
  --task instruction

# 2. Use examples/config_instruction.yaml

# 3. Train
python finetune.py --config examples/config_instruction.yaml

# 4. Test
python inference.py --model-path ./output/instruction-model --model-type causal_lm --use-peft
```

## Troubleshooting Quick Fixes

### Out of Memory
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16

model:
  load_in_4bit: true

lora:
  use_lora: true
```

### Slow Training
```yaml
training:
  fp16: true
  batch_size: 8  # increase if memory allows
```

### Poor Results
- Increase training data
- Train more epochs: `num_epochs: 5`
- Adjust learning rate: `learning_rate: 5.0e-5`
- Use larger model

## Makefile Commands

```bash
make install    # Install dependencies
make train      # Start training
make inference  # Run inference
make evaluate   # Evaluate model
make clean      # Clean outputs
```

## File Formats

### Text Generation
```json
{"text": "Complete text for training"}
```

### Classification
```json
{"text": "Text to classify", "label": 0}
```

### Instruction
```json
{"instruction": "Question", "input": "Context", "output": "Answer"}
```

### Conversation
```json
{"user": "User message", "assistant": "Bot response"}
```

## Useful Paths

- Training output: `./output/`
- Logs: `./logs/`
- Sample data: `./examples/`
- Configs: `./examples/*.yaml`

## Quick Checks

```bash
# Verify setup
python test_setup.py

# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# List files
ls -lh *.py *.yaml
```

## Help

```bash
# Script help
python finetune.py --help
python prepare_data.py --help
python inference.py --help
python evaluate.py --help

# Documentation
cat README.md
cat TUTORIAL.md
cat FAQ.md
```

## Links

- Documentation: README.md, TUTORIAL.md, FAQ.md
- Examples: examples/
- GitHub: https://github.com/Fadil369/fine-tune
