"""
Inference script for fine-tuned models
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel


def load_model(model_path: str, model_type: str = "causal_lm", use_peft: bool = False):
    """Load fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if model_type == "causal_lm":
        if use_peft:
            base_model = AutoModelForCausalLM.from_pretrained(model_path)
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
    elif model_type == "seq2seq":
        if use_peft:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
):
    """Generate text from prompt"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts


def classify_text(model, tokenizer, text: str):
    """Classify text"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    return predicted_class, confidence


def interactive_mode(model, tokenizer, model_type: str):
    """Interactive mode for testing the model"""
    print("\n" + "="*50)
    print("Interactive Mode - Enter 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not prompt:
            continue
        
        if model_type == "classification":
            predicted_class, confidence = classify_text(model, tokenizer, prompt)
            print(f"\nPredicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
        else:
            generated = generate_text(model, tokenizer, prompt)
            print(f"\nGenerated: {generated[0]}")


def batch_inference(
    model,
    tokenizer,
    input_file: str,
    output_file: str,
    model_type: str = "causal_lm",
):
    """Run inference on a batch of inputs from a file"""
    import json
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get('text', data.get('prompt', ''))
            
            if model_type == "classification":
                predicted_class, confidence = classify_text(model, tokenizer, prompt)
                results.append({
                    'input': prompt,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                })
            else:
                generated = generate_text(model, tokenizer, prompt)
                results.append({
                    'input': prompt,
                    'generated': generated[0],
                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Processed {len(results)} examples and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned models")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="causal_lm",
        choices=["causal_lm", "seq2seq", "classification"],
        help="Type of model",
    )
    parser.add_argument(
        "--use-peft",
        action="store_true",
        help="Whether the model uses PEFT/LoRA",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "batch"],
        help="Inference mode",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for batch inference",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for batch inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.model_type, args.use_peft)
    
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, args.model_type)
    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            print("Error: --input-file and --output-file are required for batch mode")
            return
        batch_inference(model, tokenizer, args.input_file, args.output_file, args.model_type)
    elif args.prompt:
        if args.model_type == "classification":
            predicted_class, confidence = classify_text(model, tokenizer, args.prompt)
            print(f"\nPredicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
        else:
            generated = generate_text(
                model, tokenizer, args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print(f"\nGenerated: {generated[0]}")


if __name__ == "__main__":
    main()
