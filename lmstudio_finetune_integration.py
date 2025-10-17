"""
LM Studio Integration for Fine-Tuning Framework
Connects LM Studio's local inference capabilities with fine-tuning workflow
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# Add LM Studio integration to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lmstudio_integration import (
        LMStudioClient,
        AuditLogger,
        MCPManager,
        ToolRegistry,
        PLUGINS_DIR,
        CONFIG_DIR
    )
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    print("LM Studio integration not available. Copy enhanced-cli.py to lmstudio_integration.py")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuneAssistant:
    """
    AI Assistant for fine-tuning workflow using LM Studio
    Provides intelligent guidance, data analysis, and optimization suggestions
    """
    
    def __init__(
        self,
        lmstudio_url: str = "http://localhost:1234/v1",
        config_dir: Path = CONFIG_DIR
    ):
        if not LMSTUDIO_AVAILABLE:
            raise ImportError("LM Studio integration not available")
            
        self.audit_logger = AuditLogger()
        self.llm = LMStudioClient(lmstudio_url, self.audit_logger)
        self.mcp_manager = MCPManager(config_dir, self.audit_logger)
        self.tool_registry = ToolRegistry(PLUGINS_DIR, self.audit_logger)
        self.conversation_history = []
        
    async def initialize(self):
        """Initialize the assistant"""
        logger.info("Initializing Fine-Tune Assistant...")
        await self.mcp_manager.load_mcp_config()
        await self.tool_registry.discover_plugins()
        logger.info("Assistant initialized successfully")
        
    async def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze dataset using LLM intelligence
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Analysis results with recommendations
        """
        # Read sample of dataset
        import pandas as pd
        
        try:
            if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
                with open(dataset_path, 'r') as f:
                    lines = [json.loads(line) for line in f.readlines()[:10]]
                sample = json.dumps(lines, indent=2)
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path, nrows=10)
                sample = df.to_string()
            else:
                return {"error": "Unsupported file format"}
                
            # Ask LLM to analyze
            prompt = f"""Analyze this training dataset sample and provide:
1. Data quality assessment
2. Potential issues (bias, noise, formatting)
3. Recommendations for preprocessing
4. Suggested hyperparameters based on data characteristics
5. Estimated training time and resources needed

Dataset sample:
{sample}

Provide structured analysis as JSON."""

            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            # Try to parse as JSON, fallback to text
            try:
                analysis = json.loads(response)
            except:
                analysis = {"raw_analysis": response}
                
            return analysis
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            return {"error": str(e)}
    
    async def suggest_hyperparameters(
        self,
        model_name: str,
        dataset_size: int,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Get hyperparameter suggestions from LLM
        
        Args:
            model_name: Name of the base model
            dataset_size: Number of training samples
            task_type: Type of task (classification, generation, etc.)
            
        Returns:
            Hyperparameter recommendations
        """
        prompt = f"""As an ML expert, suggest optimal hyperparameters for fine-tuning:

Model: {model_name}
Dataset size: {dataset_size} samples
Task type: {task_type}

Provide recommendations for:
1. Learning rate
2. Batch size
3. Number of epochs
4. LoRA parameters (r, alpha, dropout)
5. Warmup steps
6. Weight decay

Return as JSON with explanations."""

        response = await self.llm.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        
        try:
            suggestions = json.loads(response)
        except:
            suggestions = {"raw_response": response}
            
        return suggestions
    
    async def diagnose_training_issues(
        self,
        log_history: List[Dict],
        config: Dict
    ) -> Dict[str, Any]:
        """
        Diagnose training issues using LLM analysis
        
        Args:
            log_history: Training log history
            config: Training configuration
            
        Returns:
            Diagnosis and recommendations
        """
        # Extract key metrics
        metrics_summary = {
            "initial_loss": log_history[0].get("loss") if log_history else None,
            "final_loss": log_history[-1].get("loss") if log_history else None,
            "loss_trend": "improving" if (
                log_history and len(log_history) > 1 and
                log_history[-1].get("loss", float('inf')) < log_history[0].get("loss", float('inf'))
            ) else "not improving",
            "total_steps": len(log_history)
        }
        
        prompt = f"""Analyze these training metrics and diagnose issues:

Training Summary:
{json.dumps(metrics_summary, indent=2)}

Configuration:
{json.dumps(config, indent=2)}

Sample log entries:
{json.dumps(log_history[-5:], indent=2)}

Identify:
1. Training issues (overfitting, underfitting, instability)
2. Configuration problems
3. Suggested fixes
4. Whether to continue or restart training

Provide structured diagnosis as JSON."""

        response = await self.llm.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.3)
        
        try:
            diagnosis = json.loads(response)
        except:
            diagnosis = {"raw_diagnosis": response}
            
        return diagnosis
    
    async def generate_training_report(
        self,
        output_dir: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive training report using LLM
        
        Args:
            output_dir: Directory with training outputs
            metrics: Training metrics
            
        Returns:
            Formatted report as markdown
        """
        prompt = f"""Generate a comprehensive training report for:

Output Directory: {output_dir}
Metrics: {json.dumps(metrics, indent=2)}

Include:
1. Executive summary
2. Training performance
3. Model quality assessment
4. Recommendations for improvement
5. Next steps

Format as markdown with sections, tables, and bullet points."""

        response = await self.llm.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.5, max_tokens=3000)
        
        return response
    
    async def interactive_session(self):
        """Start interactive fine-tuning assistant session"""
        print("\n" + "="*80)
        print("Fine-Tuning AI Assistant")
        print("Ask questions about fine-tuning, get suggestions, analyze results")
        print("Type 'exit' to quit, 'help' for commands")
        print("="*80 + "\n")
        
        commands = {
            "analyze": "Analyze a dataset: analyze <path>",
            "suggest": "Get hyperparameter suggestions: suggest <model> <size> <task>",
            "diagnose": "Diagnose training issues: diagnose <output_dir>",
            "report": "Generate training report: report <output_dir>",
            "help": "Show available commands",
            "exit": "Exit the assistant"
        }
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                    
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    for cmd, desc in commands.items():
                        print(f"  {cmd}: {desc}")
                    continue
                    
                # Handle specific commands
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "analyze" and args:
                    result = await self.analyze_dataset(args)
                    print(f"\nAnalysis:\n{json.dumps(result, indent=2)}")
                    
                elif command == "suggest" and args:
                    # Parse: model size task
                    params = args.split()
                    if len(params) >= 3:
                        result = await self.suggest_hyperparameters(
                            params[0], int(params[1]), params[2]
                        )
                        print(f"\nSuggestions:\n{json.dumps(result, indent=2)}")
                    else:
                        print("Usage: suggest <model> <dataset_size> <task_type>")
                        
                elif command == "diagnose" and args:
                    # Load trainer state
                    state_path = Path(args) / "trainer_state.json"
                    if state_path.exists():
                        with open(state_path, 'r') as f:
                            state = json.load(f)
                            
                        # Load config
                        config_path = Path(args).parent / "config.yaml"
                        if config_path.exists():
                            import yaml
                            with open(config_path, 'r') as f:
                                config = yaml.safe_load(f)
                        else:
                            config = {}
                            
                        result = await self.diagnose_training_issues(
                            state.get("log_history", []),
                            config
                        )
                        print(f"\nDiagnosis:\n{json.dumps(result, indent=2)}")
                    else:
                        print(f"No trainer_state.json found in {args}")
                        
                elif command == "report" and args:
                    # Load metrics
                    state_path = Path(args) / "trainer_state.json"
                    if state_path.exists():
                        with open(state_path, 'r') as f:
                            state = json.load(f)
                        metrics = state.get("log_history", [{}])[-1]
                        report = await self.generate_training_report(args, metrics)
                        print(f"\n{report}")
                    else:
                        print(f"No trainer_state.json found in {args}")
                        
                else:
                    # General question - use LLM
                    self.conversation_history.append({"role": "user", "content": user_input})
                    
                    response = await self.llm.chat([
                        {
                            "role": "system",
                            "content": "You are an expert ML engineer specializing in fine-tuning language models. "
                                      "Provide practical, actionable advice about fine-tuning, hyperparameters, "
                                      "training strategies, and troubleshooting."
                        }
                    ] + self.conversation_history)
                    
                    self.conversation_history.append({"role": "assistant", "content": response})
                    print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-Tuning AI Assistant")
    parser.add_argument(
        "--mode",
        choices=["interactive", "analyze", "suggest", "diagnose", "report"],
        default="interactive",
        help="Operation mode"
    )
    parser.add_argument("--dataset", help="Dataset path (for analyze)")
    parser.add_argument("--model", help="Model name (for suggest)")
    parser.add_argument("--size", type=int, help="Dataset size (for suggest)")
    parser.add_argument("--task", help="Task type (for suggest)")
    parser.add_argument("--output-dir", help="Output directory (for diagnose/report)")
    
    args = parser.parse_args()
    
    try:
        assistant = FineTuneAssistant()
        await assistant.initialize()
        
        if args.mode == "interactive":
            await assistant.interactive_session()
            
        elif args.mode == "analyze" and args.dataset:
            result = await assistant.analyze_dataset(args.dataset)
            print(json.dumps(result, indent=2))
            
        elif args.mode == "suggest" and args.model and args.size and args.task:
            result = await assistant.suggest_hyperparameters(args.model, args.size, args.task)
            print(json.dumps(result, indent=2))
            
        elif args.mode == "diagnose" and args.output_dir:
            state_path = Path(args.output_dir) / "trainer_state.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                result = await assistant.diagnose_training_issues(
                    state.get("log_history", []),
                    {}
                )
                print(json.dumps(result, indent=2))
            else:
                print(f"No trainer_state.json found in {args.output_dir}")
                
        elif args.mode == "report" and args.output_dir:
            state_path = Path(args.output_dir) / "trainer_state.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                metrics = state.get("log_history", [{}])[-1]
                report = await assistant.generate_training_report(args.output_dir, metrics)
                print(report)
            else:
                print(f"No trainer_state.json found in {args.output_dir}")
                
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {e}")
        print(f"\nError: {e}")
        print("\nMake sure LM Studio is running: http://localhost:1234")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
