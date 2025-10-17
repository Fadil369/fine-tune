"""
Fine-Tuning Manager Plugin for LM Studio Integration
Provides tools to start, monitor, and control fine-tuning jobs
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class FineTuneManagerPlugin:
    """Plugin for managing fine-tuning operations"""
    
    name = "finetune_manager"
    description = "Start, stop, and monitor fine-tuning training jobs"
    
    def __init__(self):
        self.workspace_dir = Path(__file__).parent.parent
        
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute fine-tuning management actions
        
        Args:
            action: Action to perform (start, stop, status, list)
            **kwargs: Action-specific parameters
            
        Returns:
            Result dictionary
        """
        if action == "start":
            return self.start_training(**kwargs)
        elif action == "stop":
            return self.stop_training(**kwargs)
        elif action == "status":
            return self.get_status(**kwargs)
        elif action == "list":
            return self.list_runs()
        else:
            return {"error": f"Unknown action: {action}"}
    
    def start_training(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        background: bool = True
    ) -> Dict[str, Any]:
        """
        Start a new fine-tuning job
        
        Args:
            config_path: Path to training config YAML
            output_dir: Optional output directory
            background: Run in background
            
        Returns:
            Job information
        """
        config_file = Path(config_path)
        if not config_file.exists():
            return {"error": f"Config file not found: {config_path}"}
        
        # Build command
        cmd = ["python", "finetune.py", "--config", str(config_file)]
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        
        try:
            if background:
                # Start in background
                process = subprocess.Popen(
                    cmd,
                    cwd=self.workspace_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                return {
                    "status": "started",
                    "pid": process.pid,
                    "config": str(config_file),
                    "output_dir": output_dir
                }
            else:
                # Run synchronously
                result = subprocess.run(
                    cmd,
                    cwd=self.workspace_dir,
                    capture_output=True,
                    text=True
                )
                return {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout[-500:],  # Last 500 chars
                    "stderr": result.stderr[-500:]
                }
        except Exception as e:
            return {"error": str(e)}
    
    def stop_training(self, pid: int) -> Dict[str, Any]:
        """
        Stop a running training job
        
        Args:
            pid: Process ID to stop
            
        Returns:
            Stop status
        """
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
            return {"status": "stopped", "pid": pid}
        except ProcessLookupError:
            return {"error": f"Process {pid} not found"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self, output_dir: str) -> Dict[str, Any]:
        """
        Get status of a training run
        
        Args:
            output_dir: Training output directory
            
        Returns:
            Training status and metrics
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            return {"error": f"Output directory not found: {output_dir}"}
        
        # Read trainer state
        state_file = output_path / "trainer_state.json"
        if not state_file.exists():
            return {
                "status": "not_started",
                "output_dir": str(output_path)
            }
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            log_history = state.get("log_history", [])
            latest_metrics = log_history[-1] if log_history else {}
            
            return {
                "status": "running" if state.get("global_step", 0) < state.get("max_steps", 0) else "completed",
                "global_step": state.get("global_step", 0),
                "max_steps": state.get("max_steps", 0),
                "epoch": state.get("epoch", 0),
                "best_metric": state.get("best_metric"),
                "latest_metrics": latest_metrics,
                "output_dir": str(output_path)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_runs(self) -> Dict[str, Any]:
        """
        List all training runs in outputs directory
        
        Returns:
            List of training runs
        """
        outputs_dir = self.workspace_dir / "outputs"
        if not outputs_dir.exists():
            return {"runs": []}
        
        runs = []
        for run_dir in outputs_dir.iterdir():
            if run_dir.is_dir():
                state_file = run_dir / "trainer_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        runs.append({
                            "name": run_dir.name,
                            "path": str(run_dir),
                            "global_step": state.get("global_step", 0),
                            "epoch": state.get("epoch", 0),
                            "best_metric": state.get("best_metric")
                        })
                    except:
                        pass
        
        return {"runs": runs, "total": len(runs)}


# For LM Studio plugin system
def get_plugin():
    """Factory function for plugin loading"""
    return FineTuneManagerPlugin()
