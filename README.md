# Fine-Tune Custom Models 🚀

A **production-ready**, flexible and easy-to-use framework for fine-tuning machine learning models on custom datasets. Supports various model types including causal language models, sequence-to-sequence models, and classification models.

## ✨ Features

### Core Capabilities
- 🚀 **Easy to Use**: Simple configuration-based fine-tuning
- 🎯 **Multiple Model Types**: Support for causal LM, seq2seq, and classification
- 💾 **Data Flexibility**: Works with CSV, JSON, TXT files or HuggingFace datasets
- ⚡ **Efficient Training**: Support for LoRA, 4-bit/8-bit quantization
- 📊 **Monitoring**: TensorBoard and Weights & Biases integration
- 🔧 **Customizable**: Extensive configuration options
- 🎓 **Examples**: Pre-configured examples for common use cases

### 🆕 Advanced Features (NEW!)
- 🎯 **Hyperparameter Optimization**: Grid search, random search, and Bayesian optimization with Optuna
- 📊 **Advanced Visualizations**: Training curves, gradient analysis, model comparison dashboards
- 🚀 **Production Deployment**: FastAPI REST API, Gradio UI, Docker containerization
- 📈 **Real-time Monitoring**: Live training metrics and interactive dashboards
- 🔄 **Model Comparison**: Side-by-side performance analysis
- 🌐 **Cloud-Ready**: AWS, Azure, GCP deployment templates
- 🤖 **AI-Assisted Fine-Tuning**: LM Studio integration for intelligent guidance (v2.1.0+)

> **See [ENHANCEMENTS.md](ENHANCEMENTS.md) for detailed documentation of new features!**  
> **See [LMSTUDIO_INTEGRATION.md](LMSTUDIO_INTEGRATION.md) for AI assistant guide!**

## 🎯 Quick Start Examples

### Standard Training
```bash
python finetune.py --config config.yaml
```

### Hyperparameter Optimization
```bash
python hyperparameter_tuning.py --config config.yaml --method optuna --n-trials 50
```

### Create Visualizations
```bash
python visualization_tools.py --output-dir ./output --dashboard
```

### Deploy to Production
```bash
python deployment_utils.py --output-dir ./deployment
cd deployment && python api_server.py
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Fadil369/fine-tune.git
cd fine-tune

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Create a JSON file with your training data:

```json
{"text": "Your training text here..."}
{"text": "Another training example..."}
```

Or use the data preparation utility:

```bash
python prepare_data.py \
  --input your_data.csv \
  --output prepared_data.json \
  --task instruction \
  --instruction-column question \
  --output-column answer
```

### 2. Configure Your Training

Edit `config.yaml` or create a custom configuration file:

```yaml
model:
  name: "gpt2"
  type: "causal_lm"

training:
  output_dir: "./output"
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5

data:
  train_file: "your_data.json"
  max_seq_length: 512
  text_column: "text"

lora:
  use_lora: true
  r: 8
  lora_alpha: 32
```

### 3. Start Fine-Tuning

```bash
python finetune.py --config config.yaml
```

### 4. Use Your Fine-Tuned Model

```bash
# Interactive mode
python inference.py --model-path ./output --model-type causal_lm

# Single prompt
python inference.py \
  --model-path ./output \
  --model-type causal_lm \
  --prompt "Your prompt here"

# Batch inference
python inference.py \
  --model-path ./output \
  --model-type causal_lm \
  --mode batch \
  --input-file prompts.json \
  --output-file results.json
```

## Usage Examples

### Text Generation / Instruction Following

```bash
# 1. Prepare instruction data
python prepare_data.py \
  --input raw_data.csv \
  --output instruction_data.json \
  --task instruction

# 2. Fine-tune the model
python finetune.py --config examples/config_instruction.yaml

# 3. Test the model
python inference.py \
  --model-path ./output/instruction-model \
  --model-type causal_lm \
  --use-peft
```

### Text Classification

```bash
# 1. Prepare classification data
python prepare_data.py \
  --input reviews.csv \
  --output classification_data.json \
  --task classification \
  --text-column review_text \
  --label-column sentiment

# 2. Fine-tune the model
python finetune.py --config examples/config_classification.yaml

# 3. Classify new texts
python inference.py \
  --model-path ./output/classification-model \
  --model-type classification \
  --prompt "This product is amazing!"
```

### Conversation / Chat Model

```bash
# 1. Prepare conversation data
python prepare_data.py \
  --input conversations.csv \
  --output conversation_data.json \
  --task conversation \
  --user-column user_message \
  --assistant-column bot_response

# 2. Fine-tune the model
python finetune.py --config config.yaml

# 3. Interactive chat
python inference.py \
  --model-path ./output \
  --model-type causal_lm
```

## Configuration Reference

### Model Configuration

```yaml
model:
  name: "model-name"           # HuggingFace model name or local path
  type: "causal_lm"            # causal_lm, seq2seq, or classification
  num_labels: 2                # For classification models
  load_in_8bit: false          # Enable 8-bit quantization
  load_in_4bit: false          # Enable 4-bit quantization
```

### Training Configuration

```yaml
training:
  output_dir: "./output"              # Where to save the model
  num_epochs: 3                       # Number of training epochs
  batch_size: 4                       # Per-device batch size
  gradient_accumulation_steps: 4      # Gradient accumulation
  learning_rate: 2.0e-5              # Learning rate
  warmup_steps: 100                   # Warmup steps
  max_steps: -1                       # Max steps (-1 for epoch-based)
  logging_steps: 10                   # Log every N steps
  save_steps: 500                     # Save checkpoint every N steps
  eval_steps: 500                     # Evaluate every N steps
  save_total_limit: 3                 # Keep only N checkpoints
  fp16: false                         # Enable FP16 training
  bf16: false                         # Enable BF16 training
```

### Data Configuration

```yaml
data:
  dataset_name: null                # HuggingFace dataset name
  train_file: "data.json"           # Local training file
  validation_file: null             # Local validation file
  test_file: null                   # Local test file
  max_seq_length: 512               # Maximum sequence length
  text_column: "text"               # Text column name
  label_column: null                # Label column (for classification)
```

### LoRA Configuration

```yaml
lora:
  use_lora: true                    # Enable LoRA
  r: 8                              # LoRA rank
  lora_alpha: 32                    # LoRA alpha
  lora_dropout: 0.1                 # LoRA dropout
  target_modules: ["q_proj", "v_proj"]  # Modules to apply LoRA
  bias: "none"                      # Bias training strategy
  task_type: "CAUSAL_LM"           # Task type
```

## Data Preparation Utilities

### Split Data

```bash
python prepare_data.py \
  --input data.json \
  --output split_data.json \
  --task split
```

This creates `data_train.json`, `data_val.json`, and `data_test.json`.

### Supported Data Formats

#### Text Generation
```json
{"text": "Complete text for language modeling"}
```

#### Classification
```json
{"text": "Text to classify", "label": 0}
```

#### Instruction Following
```json
{
  "instruction": "What is AI?",
  "input": "",
  "output": "AI is artificial intelligence..."
}
```

#### Conversation
```json
{
  "user": "Hello, how are you?",
  "assistant": "I'm doing well, thank you!"
}
```

## Advanced Features

### Using Pre-trained Models from HuggingFace

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  type: "causal_lm"
```

### Memory-Efficient Training with LoRA

```yaml
lora:
  use_lora: true
  r: 8
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### 4-bit Quantization for Large Models

```yaml
model:
  load_in_4bit: true

lora:
  use_lora: true
```

### Weights & Biases Integration

```yaml
logging:
  use_wandb: true
  wandb_project: "my-finetuning-project"
  wandb_run_name: "experiment-1"
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs
```

### Weights & Biases

Enable in config and view at wandb.ai

## Tips for Better Fine-Tuning

1. **Start Small**: Begin with a smaller model and dataset to validate your pipeline
2. **Data Quality**: Clean, high-quality data is more important than quantity
3. **Learning Rate**: Start with 2e-5 to 5e-5 for most models
4. **Batch Size**: Adjust based on your GPU memory
5. **LoRA for Large Models**: Use LoRA for models >1B parameters
6. **Monitor Overfitting**: Use validation set and early stopping
7. **Experiment**: Try different hyperparameters and configurations

## Troubleshooting

### Out of Memory Error

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Enable `load_in_8bit` or `load_in_4bit`
- Use `lora.use_lora: true`
- Reduce `max_seq_length`

### Poor Performance

- Increase training data
- Train for more epochs
- Adjust learning rate
- Check data quality and format
- Try different model architectures

### Slow Training

- Enable `fp16: true` (for NVIDIA GPUs)
- Enable `bf16: true` (for newer GPUs)
- Increase `batch_size` if memory allows
- Use multiple GPUs with `accelerate`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace Transformers
- PEFT Library
- PyTorch Team

## Support

For issues and questions, please open an issue on GitHub.