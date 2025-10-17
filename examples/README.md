# Example Configuration for Different Use Cases

This directory contains example configurations for various fine-tuning scenarios.

## 1. Text Generation (Causal LM)

Use `config_generation.yaml` for text generation tasks like:
- Story/content generation
- Code completion
- Dialogue systems

## 2. Text Classification

Use `config_classification.yaml` for classification tasks like:
- Sentiment analysis
- Topic categorization
- Intent detection

## 3. Instruction Following

Use `config_instruction.yaml` for instruction-following models like:
- Question answering
- Task-specific assistants
- Custom chatbots

## Data Formats

### Text Generation
```json
{"text": "Your training text here..."}
```

### Classification
```json
{"text": "Text to classify", "label": 0}
```

### Instruction Following
```json
{"instruction": "Question or instruction", "input": "Optional context", "output": "Expected response"}
```
