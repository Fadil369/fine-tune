"""
Hyperparameter Assistant Plugin
Uses LLM to suggest optimal hyperparameters based on dataset and task
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class HyperparameterAssistantPlugin:
    """Plugin for AI-assisted hyperparameter selection"""
    
    name = "hyperparameter_assistant"
    description = "Get intelligent hyperparameter recommendations using LLM analysis"
    
    def __init__(self):
        self.workspace_dir = Path(__file__).parent.parent
        
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute hyperparameter assistant actions
        
        Args:
            action: Action to perform (suggest, optimize, compare)
            **kwargs: Action-specific parameters
            
        Returns:
            Result dictionary
        """
        if action == "suggest":
            return self.suggest_hyperparameters(**kwargs)
        elif action == "analyze_config":
            return self.analyze_config(**kwargs)
        elif action == "optimize":
            return self.run_optimization(**kwargs)
        else:
            return {"error": f"Unknown action: {action}"}
    
    def suggest_hyperparameters(
        self,
        model_name: str,
        dataset_size: int,
        task_type: str = "text_generation",
        gpu_memory_gb: int = 24
    ) -> Dict[str, Any]:
        """
        Generate hyperparameter suggestions based on model and dataset
        
        Args:
            model_name: Name of the base model
            dataset_size: Number of training samples
            task_type: Type of task (text_generation, classification, etc.)
            gpu_memory_gb: Available GPU memory
            
        Returns:
            Suggested hyperparameters
        """
        # Rule-based suggestions (can be enhanced with LLM)
        suggestions = {
            "learning_rate": self._suggest_learning_rate(model_name, dataset_size),
            "batch_size": self._suggest_batch_size(gpu_memory_gb, model_name),
            "num_epochs": self._suggest_epochs(dataset_size),
            "lora_r": 8 if "7b" in model_name.lower() else 16,
            "lora_alpha": 16 if "7b" in model_name.lower() else 32,
            "lora_dropout": 0.05,
            "warmup_steps": min(500, dataset_size // 10),
            "weight_decay": 0.01,
            "gradient_accumulation_steps": self._suggest_grad_accumulation(gpu_memory_gb),
            "reasoning": {
                "learning_rate": "Conservative LR for stability with LoRA",
                "batch_size": f"Optimized for {gpu_memory_gb}GB GPU memory",
                "num_epochs": "Balanced to prevent overfitting on small datasets",
                "lora_r": "Rank appropriate for model size",
                "warmup_steps": "10% of dataset for smooth learning"
            }
        }
        
        return suggestions
    
    def _suggest_learning_rate(self, model_name: str, dataset_size: int) -> float:
        """Suggest learning rate based on model size and dataset"""
        # Smaller LR for larger models
        if "70b" in model_name.lower() or "65b" in model_name.lower():
            return 1e-5
        elif "13b" in model_name.lower() or "20b" in model_name.lower():
            return 2e-5
        else:  # 7B and smaller
            return 3e-4
    
    def _suggest_batch_size(self, gpu_memory_gb: int, model_name: str) -> int:
        """Suggest batch size based on GPU memory"""
        if "70b" in model_name.lower():
            return 1 if gpu_memory_gb < 48 else 2
        elif "13b" in model_name.lower() or "20b" in model_name.lower():
            return 2 if gpu_memory_gb < 24 else 4
        else:  # 7B and smaller
            return 4 if gpu_memory_gb < 16 else 8
    
    def _suggest_epochs(self, dataset_size: int) -> int:
        """Suggest number of epochs based on dataset size"""
        if dataset_size < 100:
            return 10
        elif dataset_size < 1000:
            return 5
        elif dataset_size < 10000:
            return 3
        else:
            return 1
    
    def _suggest_grad_accumulation(self, gpu_memory_gb: int) -> int:
        """Suggest gradient accumulation steps"""
        if gpu_memory_gb < 16:
            return 8
        elif gpu_memory_gb < 24:
            return 4
        else:
            return 2
    
    def analyze_config(self, config_path: str) -> Dict[str, Any]:
        """
        Analyze a training configuration for potential issues
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Analysis with warnings and recommendations
        """
        try:
            import yaml
            config_file = Path(config_path)
            
            if not config_file.exists():
                return {"error": f"Config file not found: {config_path}"}
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            issues = []
            warnings = []
            recommendations = []
            
            # Check learning rate
            lr = config.get("learning_rate", 0)
            if lr > 1e-3:
                warnings.append("Learning rate seems high - may cause instability")
            elif lr < 1e-6:
                warnings.append("Learning rate seems low - training may be very slow")
            
            # Check batch size
            batch_size = config.get("per_device_train_batch_size", 0)
            if batch_size > 16:
                warnings.append("Large batch size may require more GPU memory")
            elif batch_size < 1:
                issues.append("Batch size must be at least 1")
            
            # Check epochs
            epochs = config.get("num_train_epochs", 1)
            if epochs > 10:
                warnings.append("Many epochs may lead to overfitting")
            
            # Check LoRA parameters
            lora_r = config.get("lora_r", 8)
            lora_alpha = config.get("lora_alpha", 16)
            if lora_alpha != 2 * lora_r:
                recommendations.append("Consider setting lora_alpha = 2 * lora_r for better performance")
            
            # Check gradient accumulation
            grad_accum = config.get("gradient_accumulation_steps", 1)
            if batch_size * grad_accum < 8:
                recommendations.append("Effective batch size is small - consider increasing gradient accumulation")
            
            return {
                "status": "ok" if not issues else "has_issues",
                "issues": issues,
                "warnings": warnings,
                "recommendations": recommendations,
                "config_path": str(config_path)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_optimization(
        self,
        config_path: str,
        method: str = "optuna",
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            config_path: Base config file
            method: Optimization method (optuna, random, grid)
            n_trials: Number of trials
            
        Returns:
            Optimization command to run
        """
        try:
            from hyperparameter_tuning import optuna_optimize
            
            return {
                "status": "ready",
                "command": f"python hyperparameter_tuning.py --config {config_path} --method {method} --trials {n_trials}",
                "message": f"Run {method} optimization with {n_trials} trials"
            }
        except ImportError:
            return {"error": "hyperparameter_tuning.py not found in workspace"}


# For LM Studio plugin system
def get_plugin():
    """Factory function for plugin loading"""
    return HyperparameterAssistantPlugin()
