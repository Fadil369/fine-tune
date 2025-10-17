"""
Fine-tuning script for custom models
Supports various model types and training configurations
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load model and tokenizer based on configuration"""
    model_name = config['model']['name']
    model_type = config['model']['type']
    
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model based on type
    load_kwargs = {}
    if config['model'].get('load_in_8bit', False):
        load_kwargs['load_in_8bit'] = True
    if config['model'].get('load_in_4bit', False):
        load_kwargs['load_in_4bit'] = True
    
    logger.info(f"Loading model: {model_name} (type: {model_type})")
    
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    elif model_type == "classification":
        num_labels = config['model'].get('num_labels', 2)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, **load_kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, tokenizer


def setup_lora(model, config: Dict[str, Any]):
    """Setup LoRA for parameter-efficient fine-tuning"""
    if not config['lora'].get('use_lora', False):
        return model
    
    logger.info("Setting up LoRA configuration")
    
    # Prepare model for k-bit training if quantization is used
    if config['model'].get('load_in_8bit') or config['model'].get('load_in_4bit'):
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def load_data(config: Dict[str, Any], tokenizer):
    """Load and prepare datasets"""
    data_config = config['data']
    
    # Load dataset
    if data_config.get('dataset_name'):
        logger.info(f"Loading dataset: {data_config['dataset_name']}")
        dataset = load_dataset(data_config['dataset_name'])
    elif data_config.get('train_file'):
        logger.info(f"Loading dataset from files")
        data_files = {}
        if data_config['train_file']:
            data_files['train'] = data_config['train_file']
        if data_config.get('validation_file'):
            data_files['validation'] = data_config['validation_file']
        if data_config.get('test_file'):
            data_files['test'] = data_config['test_file']
        
        # Determine file type
        file_ext = Path(data_config['train_file']).suffix
        if file_ext == '.csv':
            dataset = load_dataset('csv', data_files=data_files)
        elif file_ext == '.json':
            dataset = load_dataset('json', data_files=data_files)
        elif file_ext == '.txt':
            dataset = load_dataset('text', data_files=data_files)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    else:
        raise ValueError("Either dataset_name or train_file must be specified")
    
    # Tokenize dataset
    max_length = data_config.get('max_seq_length', 512)
    text_column = data_config.get('text_column', 'text')
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
    
    logger.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing",
    )
    
    return tokenized_dataset


def setup_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """Setup training arguments"""
    train_config = config['training']
    
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        warmup_steps=train_config.get('warmup_steps', 0),
        max_steps=train_config.get('max_steps', -1),
        logging_dir=config['logging'].get('log_dir', './logs'),
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config.get('eval_steps', 500),
        save_total_limit=train_config.get('save_total_limit', 3),
        evaluation_strategy="steps" if 'validation' in config else "no",
        fp16=train_config.get('fp16', False),
        bf16=train_config.get('bf16', False),
        weight_decay=config['optimizer'].get('weight_decay', 0.01),
        adam_beta1=config['optimizer'].get('adam_beta1', 0.9),
        adam_beta2=config['optimizer'].get('adam_beta2', 0.999),
        adam_epsilon=config['optimizer'].get('adam_epsilon', 1e-8),
        lr_scheduler_type=config['scheduler'].get('name', 'linear'),
        report_to=["tensorboard"] if config['logging'].get('use_tensorboard') else [],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
    
    return training_args


def train(config_path: str):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA if enabled
    model = setup_lora(model, config)
    
    # Load and prepare data
    dataset = load_data(config, tokenizer)
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Setup data collator
    model_type = config['model']['type']
    if model_type == "causal_lm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    elif model_type == "seq2seq":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
        )
    else:
        data_collator = None
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluate if test set is available
    if 'test' in dataset:
        logger.info("Evaluating on test set")
        metrics = trainer.evaluate(dataset['test'])
        logger.info(f"Test metrics: {metrics}")
    
    logger.info("Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune custom models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    train(args.config)


if __name__ == "__main__":
    main()
