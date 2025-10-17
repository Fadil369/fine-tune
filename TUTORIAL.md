# Fine-Tuning Tutorial

This tutorial will walk you through the complete process of fine-tuning a model using this framework.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)
6. [Evaluation](#evaluation)

## Installation

```bash
# Clone the repository
git clone https://github.com/Fadil369/fine-tune.git
cd fine-tune

# Install dependencies
pip install -r requirements.txt

# Or use the quick start script
bash quickstart.sh
```

## Data Preparation

### Option 1: Use Pre-formatted Data

Create a JSON file with your data:

```json
{"text": "Your training text here"}
{"text": "Another example"}
```

### Option 2: Convert Your Data

If you have CSV or other formats:

```bash
# For instruction-following tasks
python prepare_data.py \
  --input data.csv \
  --output prepared_data.json \
  --task instruction \
  --instruction-column question \
  --output-column answer

# For classification tasks
python prepare_data.py \
  --input reviews.csv \
  --output classification_data.json \
  --task classification \
  --text-column review_text \
  --label-column sentiment

# For conversation/chat data
python prepare_data.py \
  --input conversations.csv \
  --output chat_data.json \
  --task conversation \
  --user-column user_message \
  --assistant-column bot_response
```

### Option 3: Split Your Data

```bash
python prepare_data.py \
  --input prepared_data.json \
  --output split_data.json \
  --task split
```

This creates:
- `split_data_train.json` (80%)
- `split_data_val.json` (10%)
- `split_data_test.json` (10%)

## Configuration

Create or edit `config.yaml`:

```yaml
# Minimal configuration
model:
  name: "gpt2"
  type: "causal_lm"

training:
  output_dir: "./my-model"
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5

data:
  train_file: "prepared_data_train.json"
  validation_file: "prepared_data_val.json"
  max_seq_length: 512
  text_column: "text"

lora:
  use_lora: true
  r: 8
  lora_alpha: 32
```

### Configuration Tips

**For smaller GPUs (8GB or less):**
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16

model:
  load_in_4bit: true

lora:
  use_lora: true
```

**For faster training:**
```yaml
training:
  fp16: true  # For NVIDIA GPUs
  # or
  bf16: true  # For newer GPUs (A100, H100)
```

**For better results:**
```yaml
training:
  num_epochs: 5
  learning_rate: 5.0e-5
  warmup_steps: 500
```

## Training

### Start Training

```bash
python finetune.py --config config.yaml
```

### Monitor Training

**Using TensorBoard:**
```bash
tensorboard --logdir ./logs
```

**Using Weights & Biases:**
```yaml
# In config.yaml
logging:
  use_wandb: true
  wandb_project: "my-project"
```

### Training Output

The script will:
1. Load the model and tokenizer
2. Prepare the dataset
3. Train for the specified epochs
4. Save checkpoints periodically
5. Save the final model to `output_dir`

## Inference

### Interactive Mode

```bash
python inference.py \
  --model-path ./my-model \
  --model-type causal_lm
```

Then enter prompts interactively:
```
Prompt: What is machine learning?
Generated: Machine learning is a subset of artificial intelligence...

Prompt: quit
```

### Single Prompt

```bash
python inference.py \
  --model-path ./my-model \
  --model-type causal_lm \
  --prompt "Explain neural networks" \
  --max-length 256 \
  --temperature 0.7
```

### Batch Inference

Create `prompts.json`:
```json
{"text": "What is AI?"}
{"text": "How does deep learning work?"}
```

Run inference:
```bash
python inference.py \
  --model-path ./my-model \
  --model-type causal_lm \
  --mode batch \
  --input-file prompts.json \
  --output-file results.json
```

### Classification Inference

```bash
python inference.py \
  --model-path ./classification-model \
  --model-type classification \
  --prompt "This product is amazing!"
```

Output:
```
Predicted Class: 1
Confidence: 0.9523
```

## Evaluation

### Evaluate Your Model

```bash
# For generation models
python evaluate.py \
  --model-path ./my-model \
  --model-type causal_lm \
  --test-file prepared_data_test.json

# For classification models
python evaluate.py \
  --model-path ./classification-model \
  --model-type classification \
  --test-file classification_test.json \
  --text-column text \
  --label-column label
```

### Compare Multiple Models

```bash
python evaluate.py \
  --model-path "./model1,./model2,./model3" \
  --model-type causal_lm \
  --test-file test_data.json \
  --output-file comparison_results.json
```

## Complete Example: Instruction-Following Model

```bash
# 1. Prepare data
cat > instructions.csv << EOF
instruction,input,output
What is AI?,,"AI is artificial intelligence..."
Explain topic,Neural networks,"Neural networks are..."
EOF

# 2. Convert data
python prepare_data.py \
  --input instructions.csv \
  --output instruction_data.json \
  --task instruction

# 3. Split data
python prepare_data.py \
  --input instruction_data.json \
  --task split

# 4. Create config
cat > my_config.yaml << EOF
model:
  name: "gpt2"
  type: "causal_lm"

training:
  output_dir: "./instruction-model"
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8

data:
  train_file: "instruction_data_train.json"
  validation_file: "instruction_data_val.json"
  max_seq_length: 512
  text_column: "text"

lora:
  use_lora: true
  r: 8
EOF

# 5. Train
python finetune.py --config my_config.yaml

# 6. Test
python inference.py \
  --model-path ./instruction-model \
  --model-type causal_lm \
  --use-peft \
  --prompt "### Instruction:\nWhat is deep learning?\n\n### Response:\n"
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `load_in_4bit: true`
- Use `lora.use_lora: true`
- Reduce `max_seq_length`

### Slow Training

- Enable `fp16: true` or `bf16: true`
- Increase `batch_size` if you have memory
- Use a smaller model for testing

### Poor Results

- Increase training data
- Train for more epochs
- Try different learning rates (1e-5 to 5e-5)
- Check data quality
- Use a larger base model

### Configuration Errors

```bash
# Validate your config
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

## Best Practices

1. **Start small**: Test with a small dataset first
2. **Monitor training**: Use TensorBoard or W&B
3. **Save regularly**: Use reasonable `save_steps`
4. **Validate**: Always use a validation set
5. **Experiment**: Try different hyperparameters
6. **Document**: Keep notes on what works

## Next Steps

- Try different base models (GPT-2, BERT, T5, etc.)
- Experiment with LoRA parameters
- Use quantization for large models
- Deploy your model with FastAPI or Gradio
- Share your fine-tuned model on HuggingFace

## Need Help?

- Check the README.md for more details
- See examples/ directory for configuration examples
- Open an issue on GitHub
