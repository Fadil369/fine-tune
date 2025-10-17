"""
Advanced Visualization Tools for Model Training and Analysis
Includes training curves, attention visualization, and interactive dashboards
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_curves(
    log_history: List[Dict[str, Any]],
    output_path: str = "training_curves.png",
    metrics: Optional[List[str]] = None
):
    """
    Plot training and validation curves
    
    Args:
        log_history: Training log history from trainer
        output_path: Path to save the plot
        metrics: List of metrics to plot (default: loss and eval_loss)
    """
    if metrics is None:
        metrics = ['loss', 'eval_loss']
    
    # Extract data
    df = pd.DataFrame(log_history)
    
    # Create subplots for each metric pair
    fig, axes = plt.subplots(1, len(metrics)//2 + len(metrics)%2, figsize=(15, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2] if len(metrics) > 2 else axes[idx]
        
        if metric in df.columns:
            data = df[[metric]].dropna()
            if not data.empty:
                ax.plot(data.index, data[metric], label=metric, linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} over Training')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_loss_comparison(
    experiments: Dict[str, List[Dict[str, Any]]],
    output_path: str = "loss_comparison.png"
):
    """
    Compare loss curves across multiple experiments
    
    Args:
        experiments: Dictionary mapping experiment names to log histories
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training loss
    for name, log_history in experiments.items():
        df = pd.DataFrame(log_history)
        if 'loss' in df.columns:
            loss_data = df[['loss']].dropna()
            ax1.plot(loss_data.index, loss_data['loss'], label=name, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Validation loss
    for name, log_history in experiments.items():
        df = pd.DataFrame(log_history)
        if 'eval_loss' in df.columns:
            eval_data = df[['eval_loss']].dropna()
            ax2.plot(eval_data.index, eval_data['eval_loss'], label=name, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss comparison saved to {output_path}")


def plot_learning_rate_schedule(
    log_history: List[Dict[str, Any]],
    output_path: str = "learning_rate_schedule.png"
):
    """
    Plot learning rate schedule over training
    
    Args:
        log_history: Training log history
        output_path: Path to save the plot
    """
    df = pd.DataFrame(log_history)
    
    if 'learning_rate' in df.columns:
        lr_data = df[['learning_rate']].dropna()
        
        plt.figure(figsize=(12, 5))
        plt.plot(lr_data.index, lr_data['learning_rate'], linewidth=2, color='#2E86AB')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning rate schedule saved to {output_path}")
    else:
        print("No learning rate data found in log history")


def plot_gradient_norms(
    log_history: List[Dict[str, Any]],
    output_path: str = "gradient_norms.png"
):
    """
    Plot gradient norms over training to detect gradient issues
    
    Args:
        log_history: Training log history
        output_path: Path to save the plot
    """
    df = pd.DataFrame(log_history)
    
    grad_norm_cols = [col for col in df.columns if 'grad_norm' in col.lower()]
    
    if grad_norm_cols:
        plt.figure(figsize=(12, 5))
        
        for col in grad_norm_cols:
            data = df[[col]].dropna()
            plt.plot(data.index, data[col], label=col, linewidth=2, alpha=0.8)
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Gradient Norm', fontsize=12)
        plt.title('Gradient Norms Over Training', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gradient norms plot saved to {output_path}")
    else:
        print("No gradient norm data found in log history")


def create_training_dashboard(
    output_dir: str,
    log_history: Optional[List[Dict[str, Any]]] = None
):
    """
    Create comprehensive training dashboard with multiple visualizations
    
    Args:
        output_dir: Directory containing training outputs
        log_history: Optional log history (will load from trainer_state.json if not provided)
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Load log history if not provided
    if log_history is None:
        trainer_state_path = output_dir / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                log_history = trainer_state.get('log_history', [])
        else:
            print(f"No trainer_state.json found in {output_dir}")
            return
    
    if not log_history:
        print("No log history available")
        return
    
    print(f"Creating training dashboard in {viz_dir}")
    
    # Generate all visualizations
    plot_training_curves(log_history, str(viz_dir / "training_curves.png"))
    plot_learning_rate_schedule(log_history, str(viz_dir / "learning_rate.png"))
    plot_gradient_norms(log_history, str(viz_dir / "gradient_norms.png"))
    
    # Create summary statistics
    create_training_summary(log_history, str(viz_dir / "training_summary.txt"))
    
    print(f"\n✓ Training dashboard created in {viz_dir}")


def create_training_summary(
    log_history: List[Dict[str, Any]],
    output_path: str = "training_summary.txt"
):
    """
    Create text summary of training statistics
    
    Args:
        log_history: Training log history
        output_path: Path to save summary
    """
    df = pd.DataFrame(log_history)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Training metrics
        if 'loss' in df.columns:
            loss_data = df['loss'].dropna()
            f.write("Training Loss:\n")
            f.write(f"  Initial: {loss_data.iloc[0]:.4f}\n")
            f.write(f"  Final: {loss_data.iloc[-1]:.4f}\n")
            f.write(f"  Best: {loss_data.min():.4f}\n")
            f.write(f"  Mean: {loss_data.mean():.4f}\n")
            f.write(f"  Std: {loss_data.std():.4f}\n\n")
        
        # Validation metrics
        if 'eval_loss' in df.columns:
            eval_data = df['eval_loss'].dropna()
            if len(eval_data) > 0:
                f.write("Validation Loss:\n")
                f.write(f"  Best: {eval_data.min():.4f}\n")
                f.write(f"  Final: {eval_data.iloc[-1]:.4f}\n")
                f.write(f"  Mean: {eval_data.mean():.4f}\n")
                f.write(f"  Std: {eval_data.std():.4f}\n\n")
        
        # Learning rate
        if 'learning_rate' in df.columns:
            lr_data = df['learning_rate'].dropna()
            f.write("Learning Rate:\n")
            f.write(f"  Initial: {lr_data.iloc[0]:.2e}\n")
            f.write(f"  Final: {lr_data.iloc[-1]:.2e}\n")
            f.write(f"  Max: {lr_data.max():.2e}\n")
            f.write(f"  Min: {lr_data.min():.2e}\n\n")
        
        # Training time
        if 'epoch' in df.columns:
            epochs = df['epoch'].dropna()
            f.write(f"Total Epochs: {epochs.iloc[-1] if len(epochs) > 0 else 'N/A'}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Training summary saved to {output_path}")


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str = "model_comparison.png",
    metrics: Optional[List[str]] = None
):
    """
    Create bar chart comparing multiple models
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        output_path: Path to save the plot
        metrics: List of metrics to compare
    """
    if not results:
        print("No results to plot")
        return
    
    # Auto-detect metrics if not provided
    if metrics is None:
        all_metrics = set()
        for model_metrics in results.values():
            all_metrics.update(model_metrics.keys())
        metrics = sorted(list(all_metrics))
    
    # Prepare data
    models = list(results.keys())
    metric_data = {metric: [] for metric in metrics}
    
    for model in models:
        for metric in metrics:
            metric_data[metric].append(results[model].get(metric, 0))
    
    # Create plot
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    
    for idx, metric in enumerate(metrics):
        offset = width * idx - (width * len(metrics) / 2)
        ax.bar(x + offset, metric_data[metric], width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    output_path: str = "confusion_matrix.png"
):
    """
    Plot confusion matrix for classification tasks
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        output_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    """Command line interface for visualization tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization tools for model training")
    parser.add_argument("--output-dir", type=str, required=True, help="Training output directory")
    parser.add_argument("--dashboard", action="store_true", help="Create full dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard:
        create_training_dashboard(args.output_dir)
    else:
        print("Use --dashboard to create visualizations")


if __name__ == "__main__":
    main()
