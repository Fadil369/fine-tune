"""
Advanced Hyperparameter Optimization for Fine-Tuning
Supports Grid Search, Random Search, and Optuna optimization
"""

import os
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from transformers import TrainerCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Define the hyperparameter search space"""
    learning_rate: List[float] = field(default_factory=lambda: [1e-5, 2e-5, 5e-5, 1e-4])
    batch_size: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    gradient_accumulation_steps: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    warmup_steps: List[int] = field(default_factory=lambda: [0, 100, 500])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1])
    lora_r: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    lora_alpha: List[int] = field(default_factory=lambda: [16, 32, 64])
    lora_dropout: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    num_epochs: List[int] = field(default_factory=lambda: [1, 2, 3, 5])


class OptunaPruningCallback(TrainerCallback):
    """Callback for Optuna pruning during training"""
    
    def __init__(self, trial: optuna.Trial, monitor: str = "eval_loss"):
        self.trial = trial
        self.monitor = monitor
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Report intermediate results to Optuna"""
        if metrics is not None and self.monitor in metrics:
            self.trial.report(metrics[self.monitor], state.global_step)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def grid_search(
    base_config: Dict[str, Any],
    param_space: HyperparameterSpace,
    train_function: Callable,
    output_dir: str = "./tuning_results"
) -> Dict[str, Any]:
    """
    Perform grid search over hyperparameter space
    
    Args:
        base_config: Base configuration dictionary
        param_space: Hyperparameter search space
        train_function: Function to train model with given config
        output_dir: Directory to save results
        
    Returns:
        Best configuration found
    """
    import itertools
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    param_names = []
    param_values = []
    
    for attr_name in dir(param_space):
        if not attr_name.startswith('_'):
            attr_value = getattr(param_space, attr_name)
            if isinstance(attr_value, list) and len(attr_value) > 0:
                param_names.append(attr_name)
                param_values.append(attr_value)
    
    all_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Grid Search: Testing {len(all_combinations)} combinations")
    
    best_score = float('inf')
    best_config = None
    results = []
    
    for idx, combination in enumerate(all_combinations):
        # Create config for this combination
        trial_config = base_config.copy()
        params = dict(zip(param_names, combination))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Trial {idx+1}/{len(all_combinations)}: {params}")
        logger.info(f"{'='*80}")
        
        # Update config with trial parameters
        trial_config = update_config_with_params(trial_config, params)
        trial_config['training']['output_dir'] = f"{output_dir}/trial_{idx}"
        
        try:
            # Train model
            metrics = train_function(trial_config)
            score = metrics.get('eval_loss', metrics.get('loss', float('inf')))
            
            results.append({
                'trial': idx,
                'params': params,
                'score': score,
                'metrics': metrics
            })
            
            # Update best
            if score < best_score:
                best_score = score
                best_config = params
                logger.info(f"✓ New best score: {best_score:.4f}")
            
        except Exception as e:
            logger.error(f"Trial {idx} failed: {e}")
            results.append({
                'trial': idx,
                'params': params,
                'score': None,
                'error': str(e)
            })
    
    # Save results
    results_path = f"{output_dir}/grid_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Grid Search Complete!")
    logger.info(f"Best Score: {best_score:.4f}")
    logger.info(f"Best Params: {best_config}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"{'='*80}\n")
    
    return best_config


def random_search(
    base_config: Dict[str, Any],
    param_space: HyperparameterSpace,
    train_function: Callable,
    n_trials: int = 20,
    output_dir: str = "./tuning_results"
) -> Dict[str, Any]:
    """
    Perform random search over hyperparameter space
    
    Args:
        base_config: Base configuration dictionary
        param_space: Hyperparameter search space
        train_function: Function to train model with given config
        n_trials: Number of random trials
        output_dir: Directory to save results
        
    Returns:
        Best configuration found
    """
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Random Search: Testing {n_trials} random combinations")
    
    best_score = float('inf')
    best_config = None
    results = []
    
    for trial_idx in range(n_trials):
        # Randomly sample parameters
        params = {}
        for attr_name in dir(param_space):
            if not attr_name.startswith('_'):
                attr_value = getattr(param_space, attr_name)
                if isinstance(attr_value, list) and len(attr_value) > 0:
                    params[attr_name] = random.choice(attr_value)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Trial {trial_idx+1}/{n_trials}: {params}")
        logger.info(f"{'='*80}")
        
        # Update config
        trial_config = base_config.copy()
        trial_config = update_config_with_params(trial_config, params)
        trial_config['training']['output_dir'] = f"{output_dir}/trial_{trial_idx}"
        
        try:
            # Train model
            metrics = train_function(trial_config)
            score = metrics.get('eval_loss', metrics.get('loss', float('inf')))
            
            results.append({
                'trial': trial_idx,
                'params': params,
                'score': score,
                'metrics': metrics
            })
            
            if score < best_score:
                best_score = score
                best_config = params
                logger.info(f"✓ New best score: {best_score:.4f}")
                
        except Exception as e:
            logger.error(f"Trial {trial_idx} failed: {e}")
            results.append({
                'trial': trial_idx,
                'params': params,
                'score': None,
                'error': str(e)
            })
    
    # Save results
    results_path = f"{output_dir}/random_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Random Search Complete!")
    logger.info(f"Best Score: {best_score:.4f}")
    logger.info(f"Best Params: {best_config}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"{'='*80}\n")
    
    return best_config


def optuna_optimize(
    base_config: Dict[str, Any],
    train_function: Callable,
    n_trials: int = 50,
    output_dir: str = "./tuning_results",
    study_name: Optional[str] = None,
    use_pruning: bool = True
) -> Dict[str, Any]:
    """
    Perform Bayesian optimization using Optuna
    
    Args:
        base_config: Base configuration dictionary
        train_function: Function to train model with given config
        n_trials: Number of optimization trials
        output_dir: Directory to save results
        study_name: Name for the Optuna study
        use_pruning: Whether to use pruning for early stopping
        
    Returns:
        Best configuration found
    """
    os.makedirs(output_dir, exist_ok=True)
    
    study_name = study_name or "fine_tune_optimization"
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna"""
        
        # Sample hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [1, 2, 4, 8, 16]),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4, 8, 16]),
            'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.3),
            'num_epochs': trial.suggest_int('num_epochs', 1, 5),
        }
        
        # Add LoRA params if enabled
        if base_config.get('lora', {}).get('use_lora', False):
            params['lora_r'] = trial.suggest_categorical('lora_r', [4, 8, 16, 32, 64])
            params['lora_alpha'] = trial.suggest_categorical('lora_alpha', [16, 32, 64, 128])
            params['lora_dropout'] = trial.suggest_float('lora_dropout', 0.0, 0.2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Trial {trial.number}: {params}")
        logger.info(f"{'='*80}")
        
        # Update config
        trial_config = base_config.copy()
        trial_config = update_config_with_params(trial_config, params)
        trial_config['training']['output_dir'] = f"{output_dir}/trial_{trial.number}"
        
        try:
            # Train model
            metrics = train_function(trial_config, trial if use_pruning else None)
            score = metrics.get('eval_loss', metrics.get('loss', float('inf')))
            
            return score
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner() if use_pruning else optuna.pruners.NopPruner()
    )
    
    # Optimize
    logger.info(f"Starting Optuna optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    logger.info(f"\n{'='*80}")
    logger.info(f"Optimization Complete!")
    logger.info(f"Best Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info(f"{'='*80}\n")
    
    # Save study
    study_path = f"{output_dir}/optuna_study.pkl"
    import joblib
    joblib.dump(study, study_path)
    logger.info(f"Study saved to: {study_path}")
    
    # Generate visualizations
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(f"{output_dir}/optimization_history.png")
        
        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(f"{output_dir}/param_importances.png")
        
        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image(f"{output_dir}/parallel_coordinate.png")
        
        logger.info(f"Visualizations saved to {output_dir}/")
        
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
    
    return study.best_params


def update_config_with_params(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Update config dictionary with hyperparameters"""
    import copy
    config = copy.deepcopy(config)
    
    # Training parameters
    if 'learning_rate' in params:
        config.setdefault('training', {})['learning_rate'] = params['learning_rate']
    if 'batch_size' in params:
        config.setdefault('training', {})['batch_size'] = params['batch_size']
    if 'gradient_accumulation_steps' in params:
        config.setdefault('training', {})['gradient_accumulation_steps'] = params['gradient_accumulation_steps']
    if 'warmup_steps' in params:
        config.setdefault('training', {})['warmup_steps'] = params['warmup_steps']
    if 'weight_decay' in params:
        config.setdefault('optimizer', {})['weight_decay'] = params['weight_decay']
    if 'num_epochs' in params:
        config.setdefault('training', {})['num_epochs'] = params['num_epochs']
    
    # LoRA parameters
    if 'lora_r' in params:
        config.setdefault('lora', {})['r'] = params['lora_r']
    if 'lora_alpha' in params:
        config.setdefault('lora', {})['lora_alpha'] = params['lora_alpha']
    if 'lora_dropout' in params:
        config.setdefault('lora', {})['lora_dropout'] = params['lora_dropout']
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random", "optuna"],
        default="optuna",
        help="Optimization method"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--output-dir", type=str, default="./tuning_results")
    parser.add_argument("--study-name", type=str, help="Name for Optuna study")
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Import training function
    from finetune import train
    
    def train_wrapper(config, trial=None):
        """Wrapper for training function"""
        # Save config temporarily
        temp_config_path = f"{config['training']['output_dir']}/config.yaml"
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Train and get metrics
        try:
            train(temp_config_path)
            # Load metrics from trainer state
            import json
            state_path = f"{config['training']['output_dir']}/trainer_state.json"
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    return state.get('log_history', [{}])[-1]
            return {}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'loss': float('inf')}
    
    # Run optimization
    param_space = HyperparameterSpace()
    
    if args.method == "grid":
        best_params = grid_search(base_config, param_space, train_wrapper, args.output_dir)
    elif args.method == "random":
        best_params = random_search(base_config, param_space, train_wrapper, args.n_trials, args.output_dir)
    elif args.method == "optuna":
        best_params = optuna_optimize(base_config, train_wrapper, args.n_trials, args.output_dir, args.study_name)
    
    # Save best config
    best_config = update_config_with_params(base_config, best_params)
    best_config_path = f"{args.output_dir}/best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    
    logger.info(f"\nBest configuration saved to: {best_config_path}")


if __name__ == "__main__":
    main()
