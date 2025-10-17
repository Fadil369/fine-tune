"""
Model Evaluator Plugin
Evaluate fine-tuned models and generate comparison reports
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class ModelEvaluatorPlugin:
    """Plugin for evaluating and comparing fine-tuned models"""
    
    name = "model_evaluator"
    description = "Evaluate model performance and compare different checkpoints"
    
    def __init__(self):
        self.workspace_dir = Path(__file__).parent.parent
        
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute evaluation actions
        
        Args:
            action: Action to perform (evaluate, compare, benchmark)
            **kwargs: Action-specific parameters
            
        Returns:
            Result dictionary
        """
        if action == "evaluate":
            return self.evaluate_model(**kwargs)
        elif action == "compare":
            return self.compare_models(**kwargs)
        elif action == "list_checkpoints":
            return self.list_checkpoints(**kwargs)
        else:
            return {"error": f"Unknown action: {action}"}
    
    def evaluate_model(
        self,
        model_path: str,
        test_data: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model
        
        Args:
            model_path: Path to model checkpoint
            test_data: Optional test dataset path
            metrics: Metrics to compute (perplexity, accuracy, etc.)
            
        Returns:
            Evaluation results
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            return {"error": f"Model not found: {model_path}"}
        
        # Check for trainer state
        state_file = model_dir / "trainer_state.json"
        if not state_file.exists():
            return {"error": "No trainer_state.json found"}
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            log_history = state.get("log_history", [])
            
            # Extract evaluation metrics
            eval_results = []
            for entry in log_history:
                if "eval_loss" in entry:
                    eval_results.append({
                        "step": entry.get("step", 0),
                        "epoch": entry.get("epoch", 0),
                        "eval_loss": entry.get("eval_loss"),
                        "eval_runtime": entry.get("eval_runtime"),
                        "eval_samples_per_second": entry.get("eval_samples_per_second")
                    })
            
            # Get best metrics
            best_loss = min((r["eval_loss"] for r in eval_results), default=None)
            
            return {
                "status": "success",
                "model_path": str(model_dir),
                "evaluation_count": len(eval_results),
                "best_eval_loss": best_loss,
                "latest_eval": eval_results[-1] if eval_results else None,
                "all_evaluations": eval_results,
                "best_metric": state.get("best_metric"),
                "best_model_checkpoint": state.get("best_model_checkpoint")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def compare_models(
        self,
        model_paths: List[str],
        metric: str = "eval_loss"
    ) -> Dict[str, Any]:
        """
        Compare multiple model checkpoints
        
        Args:
            model_paths: List of model checkpoint paths
            metric: Metric to compare
            
        Returns:
            Comparison results
        """
        comparisons = []
        
        for model_path in model_paths:
            result = self.evaluate_model(model_path)
            if "error" not in result:
                comparisons.append({
                    "model": model_path,
                    "best_eval_loss": result.get("best_eval_loss"),
                    "evaluation_count": result.get("evaluation_count"),
                    "best_metric": result.get("best_metric")
                })
        
        if not comparisons:
            return {"error": "No valid models to compare"}
        
        # Sort by metric
        if metric == "eval_loss":
            comparisons.sort(key=lambda x: x.get("best_eval_loss", float('inf')))
        
        best_model = comparisons[0] if comparisons else None
        
        return {
            "status": "success",
            "comparison": comparisons,
            "best_model": best_model,
            "metric": metric,
            "total_models": len(comparisons)
        }
    
    def list_checkpoints(self, output_dir: str) -> Dict[str, Any]:
        """
        List all checkpoints in an output directory
        
        Args:
            output_dir: Training output directory
            
        Returns:
            List of checkpoints with metadata
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            return {"error": f"Output directory not found: {output_dir}"}
        
        checkpoints = []
        
        # Find all checkpoint directories
        for checkpoint_dir in output_path.glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                # Extract step number
                try:
                    step = int(checkpoint_dir.name.split("-")[1])
                except:
                    step = 0
                
                # Check for trainer state
                state_file = checkpoint_dir / "trainer_state.json"
                has_state = state_file.exists()
                
                # Get directory size
                size_mb = sum(
                    f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()
                ) / (1024 * 1024)
                
                checkpoints.append({
                    "path": str(checkpoint_dir),
                    "name": checkpoint_dir.name,
                    "step": step,
                    "has_trainer_state": has_state,
                    "size_mb": round(size_mb, 2)
                })
        
        # Sort by step
        checkpoints.sort(key=lambda x: x["step"])
        
        return {
            "status": "success",
            "output_dir": str(output_path),
            "checkpoints": checkpoints,
            "total": len(checkpoints)
        }


# For LM Studio plugin system
def get_plugin():
    """Factory function for plugin loading"""
    return ModelEvaluatorPlugin()
