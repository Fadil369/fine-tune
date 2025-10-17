"""
Data preparation utilities for fine-tuning
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any


def prepare_text_classification_data(
    input_file: str,
    output_file: str,
    text_column: str = "text",
    label_column: str = "label",
):
    """
    Prepare text classification data from CSV/JSON
    
    Args:
        input_file: Path to input file
        output_file: Path to output JSON file
        text_column: Name of column containing text
        label_column: Name of column containing labels
    """
    data = []
    
    # Read input file
    if input_file.endswith('.csv'):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'text': row[text_column],
                    'label': row[label_column],
                })
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for item in raw_data:
                data.append({
                    'text': item[text_column],
                    'label': item[label_column],
                })
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Prepared {len(data)} examples and saved to {output_file}")


def prepare_instruction_data(
    input_file: str,
    output_file: str,
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
):
    """
    Prepare instruction-following data
    
    Args:
        input_file: Path to input file
        output_file: Path to output JSON file
        instruction_column: Column with instructions
        input_column: Column with inputs
        output_column: Column with expected outputs
    """
    data = []
    
    # Read input file
    if input_file.endswith('.csv'):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instruction = row[instruction_column]
                input_text = row.get(input_column, "")
                output = row[output_column]
                
                # Format as instruction-following
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
                data.append({'text': text})
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for item in raw_data:
                instruction = item[instruction_column]
                input_text = item.get(input_column, "")
                output = item[output_column]
                
                # Format as instruction-following
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
                data.append({'text': text})
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Prepared {len(data)} instruction examples and saved to {output_file}")


def prepare_conversation_data(
    input_file: str,
    output_file: str,
    user_column: str = "user",
    assistant_column: str = "assistant",
):
    """
    Prepare conversation/chat data
    
    Args:
        input_file: Path to input file
        output_file: Path to output JSON file
        user_column: Column with user messages
        assistant_column: Column with assistant responses
    """
    data = []
    
    # Read input file
    if input_file.endswith('.csv'):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_msg = row[user_column]
                assistant_msg = row[assistant_column]
                
                text = f"User: {user_msg}\nAssistant: {assistant_msg}"
                data.append({'text': text})
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for item in raw_data:
                user_msg = item[user_column]
                assistant_msg = item[assistant_column]
                
                text = f"User: {user_msg}\nAssistant: {assistant_msg}"
                data.append({'text': text})
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Prepared {len(data)} conversation examples and saved to {output_file}")


def split_data(
    input_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Split data into train/validation/test sets
    
    Args:
        input_file: Path to input JSON file
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    import random
    
    # Read data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split indices
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save splits
    base_path = Path(input_file).stem
    base_dir = Path(input_file).parent
    
    with open(base_dir / f"{base_path}_train.json", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(base_dir / f"{base_path}_val.json", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    with open(base_dir / f"{base_path}_test.json", 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Split data into:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "instruction", "conversation", "split"],
        required=True,
        help="Data preparation task"
    )
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--instruction-column", type=str, default="instruction")
    parser.add_argument("--input-column", type=str, default="input")
    parser.add_argument("--output-column", type=str, default="output")
    parser.add_argument("--user-column", type=str, default="user")
    parser.add_argument("--assistant-column", type=str, default="assistant")
    
    args = parser.parse_args()
    
    if args.task == "classification":
        prepare_text_classification_data(
            args.input,
            args.output,
            args.text_column,
            args.label_column,
        )
    elif args.task == "instruction":
        prepare_instruction_data(
            args.input,
            args.output,
            args.instruction_column,
            args.input_column,
            args.output_column,
        )
    elif args.task == "conversation":
        prepare_conversation_data(
            args.input,
            args.output,
            args.user_column,
            args.assistant_column,
        )
    elif args.task == "split":
        split_data(args.input)


if __name__ == "__main__":
    main()
