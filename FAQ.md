# Frequently Asked Questions (FAQ)

## General Questions

### What is this project?

This is a comprehensive framework for fine-tuning machine learning models (especially language models) on custom datasets. It supports various model architectures and training configurations, making it easy to adapt pre-trained models to your specific use case.

### What models are supported?

The framework supports models from HuggingFace's Transformers library:
- **Causal Language Models**: GPT-2, GPT-Neo, LLaMA, Mistral, etc.
- **Sequence-to-Sequence Models**: T5, BART, etc.
- **Classification Models**: BERT, RoBERTa, DistilBERT, etc.

### Do I need a GPU?

While you can train on CPU, a GPU is highly recommended for:
- Faster training
- Training larger models
- Handling larger batch sizes

For reference:
- **8GB GPU**: Can fine-tune GPT-2, BERT-base with LoRA
- **16GB GPU**: Can fine-tune larger models with 4-bit quantization
- **24GB+ GPU**: Can fine-tune most models comfortably

## Installation & Setup

### How do I install dependencies?

```bash
pip install -r requirements.txt
```

Or use the quick start script:
```bash
bash quickstart.sh
```

### What Python version do I need?

Python 3.8 or higher is required.

### Do I need to install CUDA separately?

PyTorch will automatically use CUDA if available. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Preparation

### What data format should I use?

JSON lines format (`.json` or `.jsonl`):
```json
{"text": "Your training text"}
{"text": "Another example"}
```

For classification:
```json
{"text": "Text to classify", "label": 0}
```

### How much data do I need?

- **Minimum**: 100-500 examples for simple tasks
- **Recommended**: 1,000-10,000 examples
- **Optimal**: 10,000+ examples

Quality matters more than quantity!

### Can I use CSV files?

Yes! Use the data preparation utility:
```bash
python prepare_data.py --input data.csv --output data.json --task instruction
```

### How do I split my data?

```bash
python prepare_data.py --input data.json --task split
```

This creates train (80%), validation (10%), and test (10%) splits.

## Training

### How long does training take?

It depends on:
- Model size
- Dataset size
- Hardware
- Hyperparameters

Example times (on RTX 3090):
- GPT-2 (124M params) + 1K samples: ~10 minutes
- GPT-2 + 10K samples: ~1-2 hours
- Larger models with LoRA: Comparable or slightly longer

### What is LoRA and should I use it?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method:
- **Pros**: Lower memory usage, faster training, smaller checkpoints
- **Cons**: Slightly different inference setup
- **Use when**: Training large models or limited GPU memory

Enable in config:
```yaml
lora:
  use_lora: true
  r: 8
  lora_alpha: 32
```

### What learning rate should I use?

Recommended starting points:
- **Small models** (GPT-2, BERT-base): `2e-5` to `5e-5`
- **Large models**: `1e-5` to `3e-5`
- **With LoRA**: `1e-4` to `3e-4`

### My training is very slow. What can I do?

1. Enable mixed precision training:
   ```yaml
   training:
     fp16: true  # or bf16: true for newer GPUs
   ```

2. Increase batch size (if memory allows):
   ```yaml
   training:
     batch_size: 8
   ```

3. Use gradient accumulation:
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 8  # effective batch size = 16
   ```

### I'm getting out of memory errors. Help!

Try these solutions in order:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 1
   ```

2. **Increase gradient accumulation**:
   ```yaml
   training:
     gradient_accumulation_steps: 16
   ```

3. **Enable 4-bit quantization**:
   ```yaml
   model:
     load_in_4bit: true
   ```

4. **Use LoRA**:
   ```yaml
   lora:
     use_lora: true
   ```

5. **Reduce sequence length**:
   ```yaml
   data:
     max_seq_length: 256  # or even 128
   ```

### How do I monitor training?

Use TensorBoard:
```bash
tensorboard --logdir ./logs
```

Or enable Weights & Biases:
```yaml
logging:
  use_wandb: true
  wandb_project: "my-project"
```

### When should I stop training?

Watch for:
- **Validation loss** stops improving
- **Training loss** much lower than validation loss (overfitting)
- **Manual testing** shows good performance

Use early stopping or monitor metrics closely.

## Inference

### How do I use my fine-tuned model?

Interactive mode:
```bash
python inference.py --model-path ./output --model-type causal_lm
```

Single prompt:
```bash
python inference.py \
  --model-path ./output \
  --model-type causal_lm \
  --prompt "Your prompt here"
```

### Can I deploy my model?

Yes! Your fine-tuned model is a standard HuggingFace model. You can:
- Load it with `transformers` library
- Deploy with FastAPI, Flask, or Gradio
- Upload to HuggingFace Hub
- Use in any Python application

### How do I adjust generation quality?

Modify generation parameters:
```bash
python inference.py \
  --model-path ./output \
  --temperature 0.7 \  # Lower = more focused, Higher = more creative
  --max-length 256     # Maximum tokens to generate
```

## Configuration

### What's the difference between epochs and steps?

- **Epoch**: One complete pass through the entire dataset
- **Step**: One forward and backward pass with a batch

If you have 1000 samples and batch size of 10:
- 1 epoch = 100 steps

### Should I use fp16 or bf16?

- **fp16**: Use for NVIDIA GPUs (V100, RTX series)
- **bf16**: Use for newer GPUs (A100, H100)
- **Neither**: Use for CPU or if you encounter numerical instability

### How many checkpoints should I keep?

```yaml
training:
  save_total_limit: 3  # Keeps only the 3 most recent checkpoints
```

Use 2-5 checkpoints to save disk space while maintaining recovery options.

## Troubleshooting

### My model generates nonsense. What's wrong?

Possible causes:
1. **Insufficient training**: Train for more epochs
2. **Learning rate too high**: Try 2e-5 instead of 5e-5
3. **Bad data quality**: Review and clean your dataset
4. **Wrong prompt format**: Ensure inference prompts match training format

### Training loss is not decreasing

1. **Learning rate too low**: Increase it
2. **Model too small**: Try a larger base model
3. **Data issues**: Check data quality and format
4. **Bugs in data**: Verify tokenization is working

### Validation loss increases while training loss decreases

This is **overfitting**:
1. **Get more data**
2. **Use early stopping**
3. **Reduce model complexity** or use LoRA
4. **Add regularization** (weight decay)

### Import errors when running scripts

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA out of memory

See "I'm getting out of memory errors" above.

## Advanced Usage

### Can I use multiple GPUs?

Yes! Use `accelerate`:
```bash
accelerate config  # Configure multi-GPU setup
accelerate launch finetune.py --config config.yaml
```

### Can I resume training from a checkpoint?

Yes! The trainer automatically resumes from the latest checkpoint in `output_dir` if available.

### How do I fine-tune a specific layer only?

Modify the model loading code to freeze layers. For most use cases, LoRA is a better option.

### Can I use custom evaluation metrics?

Yes! Modify `evaluate.py` to add your custom metrics using the `compute_metrics` function.

### How do I upload my model to HuggingFace?

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./output")
tokenizer = AutoTokenizer.from_pretrained("./output")

model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

## Best Practices

### What are the best practices for fine-tuning?

1. **Start small**: Test with a small dataset and model first
2. **Clean data**: Quality > quantity
3. **Monitor closely**: Use validation set and checkpoints
4. **Experiment**: Try different hyperparameters
5. **Document**: Keep notes on what works
6. **Version control**: Use git for your configs and data preparation scripts

### How do I choose the right base model?

Consider:
- **Task type**: Causal LM for generation, BERT for classification
- **Model size**: Balance performance vs. resources
- **Domain**: Choose models pre-trained on similar data
- **License**: Check usage restrictions

Popular choices:
- **General purpose**: GPT-2, T5, BERT
- **Code**: CodeLlama, StarCoder
- **Instruction-following**: Mistral, LLaMA-2
- **Multilingual**: mBERT, XLM-RoBERTa

## Getting Help

### Where can I get help?

1. Check this FAQ
2. Read the [TUTORIAL.md](TUTORIAL.md)
3. Review the [README.md](README.md)
4. Check example configurations in `examples/`
5. Open an issue on GitHub

### How do I report a bug?

Open a GitHub issue with:
- Description of the problem
- Steps to reproduce
- Error messages
- Your configuration file
- System info (OS, Python version, GPU)

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines!

## Additional Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
