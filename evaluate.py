"""
Evaluation script for fine-tuned models
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np


def load_model(model_path: str, model_type: str):
    """Load fine-tuned model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, tokenizer


def evaluate_classification(
    model,
    tokenizer,
    test_file: str,
    text_column: str = "text",
    label_column: str = "label",
):
    """Evaluate classification model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    predictions = []
    true_labels = []
    
    print("Evaluating classification model...")
    for item in tqdm(test_data):
        text = item[text_column]
        true_label = item[label_column]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        predictions.append(pred)
        true_labels.append(true_label)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    print("\n=== Classification Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(true_labels, predictions))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_generation(
    model,
    tokenizer,
    test_file: str,
    text_column: str = "text",
    max_length: int = 256,
):
    """Evaluate generation model with perplexity"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    total_loss = 0
    num_tokens = 0
    
    print("Evaluating generation model...")
    for item in tqdm(test_data):
        text = item[text_column]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item() * inputs['input_ids'].size(1)
            num_tokens += inputs['input_ids'].size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / num_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print("\n=== Generation Results ===")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
    }


def compare_models(
    model_paths: List[str],
    test_file: str,
    model_type: str,
):
    """Compare multiple models on the same test set"""
    results = {}
    
    for model_path in model_paths:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*50}")
        
        model, tokenizer = load_model(model_path, model_type)
        
        if model_type == "classification":
            metrics = evaluate_classification(model, tokenizer, test_file)
        else:
            metrics = evaluate_generation(model, tokenizer, test_file)
        
        results[model_path] = metrics
    
    # Print comparison
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    for model_path, metrics in results.items():
        print(f"\n{model_path}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model (or comma-separated paths for comparison)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["causal_lm", "seq2seq", "classification"],
        help="Type of model",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Test data file (JSON)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column name",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column name (for classification)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results (JSON)",
    )
    
    args = parser.parse_args()
    
    # Check if comparing multiple models
    model_paths = [p.strip() for p in args.model_path.split(',')]
    
    if len(model_paths) > 1:
        results = compare_models(model_paths, args.test_file, args.model_type)
    else:
        model, tokenizer = load_model(args.model_path, args.model_type)
        
        if args.model_type == "classification":
            results = evaluate_classification(
                model, tokenizer, args.test_file,
                args.text_column, args.label_column
            )
        else:
            results = evaluate_generation(
                model, tokenizer, args.test_file,
                args.text_column, args.max_length
            )
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
