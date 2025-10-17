# LM Studio Integration Guide

## Overview

The LM Studio integration brings AI-assisted capabilities to your fine-tuning workflow, enabling:

- **Intelligent dataset analysis** using local LLMs
- **Smart hyperparameter recommendations** based on your model and data
- **Automated training diagnostics** with actionable insights
- **Interactive AI assistant** for fine-tuning questions
- **Plugin-based extensibility** for custom workflows

## Prerequisites

1. **LM Studio** installed and running locally
   - Download from: https://lmstudio.ai
   - Start the local server (default: http://localhost:1234)
   - Load a capable model (e.g., Mistral 7B, Llama 3, etc.)

2. **Enhanced CLI** configured
   - Place `enhanced-cli.py` in workspace as `lmstudio_integration.py`
   - Configure MCP servers in `~/.lmstudio/config/mcp.json`
   - Set up plugins directory: `~/.lmstudio/plugins/`

3. **Python dependencies** installed:
   ```bash
   pip install requests rich aiohttp pyyaml pandas
   ```

## Quick Start

### 1. Interactive AI Assistant

Start an interactive session to ask questions and get guidance:

```bash
python lmstudio_finetune_integration.py --mode interactive
```

Example interactions:
```
You: What learning rate should I use for a 7B model with 1000 samples?
Assistant: For a 7B model with 1000 samples, I recommend:
- Learning rate: 2e-4 to 3e-4
- Use linear warmup for 10% of steps
- Consider gradient accumulation if GPU memory is limited
...

You: How do I prevent overfitting on a small dataset?
Assistant: For small datasets (<1000 samples), try:
1. Use fewer epochs (3-5 max)
2. Increase regularization (weight_decay=0.01)
3. Use dropout (lora_dropout=0.1)
4. Monitor validation loss closely
...
```

### 2. Dataset Analysis

Analyze your dataset before training:

```bash
python lmstudio_finetune_integration.py \
  --mode analyze \
  --dataset data/train.jsonl
```

The assistant will provide:
- Data quality assessment
- Potential issues (bias, noise, formatting problems)
- Preprocessing recommendations
- Suggested hyperparameters
- Estimated training time

Example output:
```json
{
  "data_quality": "good",
  "sample_count": 1250,
  "issues": [
    "Some samples are very short (<10 tokens)",
    "Potential class imbalance detected"
  ],
  "recommendations": [
    "Filter out samples shorter than 10 tokens",
    "Consider data augmentation for minority classes",
    "Use stratified splitting for validation"
  ],
  "suggested_hyperparameters": {
    "learning_rate": 2e-4,
    "batch_size": 4,
    "num_epochs": 3,
    "warmup_steps": 125
  }
}
```

### 3. Hyperparameter Suggestions

Get AI-powered hyperparameter recommendations:

```bash
python lmstudio_finetune_integration.py \
  --mode suggest \
  --model "mistralai/Mistral-7B-v0.1" \
  --size 1000 \
  --task text_generation
```

Returns optimized parameters based on:
- Model architecture and size
- Dataset characteristics
- Task type
- Hardware constraints

### 4. Training Diagnostics

Diagnose issues with a training run:

```bash
python lmstudio_finetune_integration.py \
  --mode diagnose \
  --output-dir outputs/run-20250118
```

The assistant analyzes:
- Loss trends (increasing, flat, decreasing)
- Learning rate schedule effectiveness
- Potential overfitting/underfitting
- Configuration issues
- Hardware utilization

Example diagnosis:
```json
{
  "status": "needs_attention",
  "issues": [
    "Loss has plateaued after epoch 2",
    "Evaluation loss diverging from training loss"
  ],
  "root_cause": "Learning rate may be too low, or model is overfitting",
  "recommendations": [
    "Increase learning rate to 3e-4",
    "Add early stopping based on eval_loss",
    "Reduce num_epochs to 3",
    "Consider increasing dropout"
  ],
  "should_restart": false,
  "should_continue": true
}
```

### 5. Training Report Generation

Generate a comprehensive training report:

```bash
python lmstudio_finetune_integration.py \
  --mode report \
  --output-dir outputs/run-20250118
```

Creates a markdown report with:
- Executive summary
- Training performance analysis
- Model quality metrics
- Visualizations (if available)
- Recommendations for improvement
- Next steps

## Plugin System

### Available Plugins

#### 1. Fine-Tune Manager (`finetune_manager`)

Manage training jobs programmatically:

```python
from plugins.finetune_manager import get_plugin

manager = get_plugin()

# Start a training job
result = manager.execute("start", 
    config_path="configs/gpt-7b.yaml",
    output_dir="outputs/experiment-1",
    background=True
)
# Returns: {"status": "started", "pid": 12345, ...}

# Check training status
status = manager.execute("status",
    output_dir="outputs/experiment-1"
)
# Returns: {"status": "running", "global_step": 250, "epoch": 1.5, ...}

# List all runs
runs = manager.execute("list")
# Returns: {"runs": [...], "total": 5}

# Stop a running job
manager.execute("stop", pid=12345)
```

#### 2. Hyperparameter Assistant (`hyperparameter_assistant`)

Get intelligent hyperparameter suggestions:

```python
from plugins.hyperparameter_assistant import get_plugin

assistant = get_plugin()

# Get suggestions
result = assistant.execute("suggest",
    model_name="mistralai/Mistral-7B-v0.1",
    dataset_size=1000,
    task_type="text_generation",
    gpu_memory_gb=24
)
# Returns detailed hyperparameter recommendations

# Analyze existing config
analysis = assistant.execute("analyze_config",
    config_path="configs/gpt-7b.yaml"
)
# Returns: {"status": "ok", "issues": [], "warnings": [...], "recommendations": [...]}
```

#### 3. Model Evaluator (`model_evaluator`)

Evaluate and compare models:

```python
from plugins.model_evaluator import get_plugin

evaluator = get_plugin()

# Evaluate a single model
result = evaluator.execute("evaluate",
    model_path="outputs/run-20250118/checkpoint-1000"
)
# Returns evaluation metrics

# Compare multiple checkpoints
comparison = evaluator.execute("compare",
    model_paths=[
        "outputs/run-20250118/checkpoint-1000",
        "outputs/run-20250118/checkpoint-2000",
        "outputs/run-20250118/checkpoint-3000"
    ],
    metric="eval_loss"
)
# Returns: {"best_model": {...}, "comparison": [...]}

# List all checkpoints
checkpoints = evaluator.execute("list_checkpoints",
    output_dir="outputs/run-20250118"
)
```

### Creating Custom Plugins

Create a new plugin in `plugins/`:

```python
# plugins/my_custom_plugin.py

class MyCustomPlugin:
    """Custom plugin description"""
    
    name = "my_custom_plugin"
    description = "What this plugin does"
    
    def execute(self, action: str, **kwargs):
        """
        Execute plugin action
        
        Args:
            action: Action to perform
            **kwargs: Action-specific parameters
            
        Returns:
            Result dictionary
        """
        if action == "my_action":
            return self.my_action(**kwargs)
        else:
            return {"error": f"Unknown action: {action}"}
    
    def my_action(self, param1, param2):
        """Implement your action"""
        return {
            "status": "success",
            "result": f"Processed {param1} and {param2}"
        }

# Factory function for plugin system
def get_plugin():
    return MyCustomPlugin()
```

The plugin will be auto-discovered by the LM Studio plugin system.

## Integration with Existing Workflow

### 1. Enhanced Training Script

Integrate AI assistant into your training script:

```python
import asyncio
from lmstudio_finetune_integration import FineTuneAssistant

async def train_with_assistant():
    # Initialize assistant
    assistant = FineTuneAssistant()
    await assistant.initialize()
    
    # Analyze dataset before training
    analysis = await assistant.analyze_dataset("data/train.jsonl")
    print(f"Dataset analysis: {analysis}")
    
    # Get hyperparameter suggestions
    suggestions = await assistant.suggest_hyperparameters(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_size=1000,
        task_type="text_generation"
    )
    
    # Apply suggestions to config
    # ... your training code ...
    
    # Diagnose if issues occur
    diagnosis = await assistant.diagnose_training_issues(
        log_history=trainer.state.log_history,
        config=training_args.to_dict()
    )
    
    if diagnosis.get("should_restart"):
        print("Training needs restart with new parameters")

asyncio.run(train_with_assistant())
```

### 2. Automated Optimization Pipeline

Combine with hyperparameter optimization:

```python
from hyperparameter_tuning import optuna_optimize
from lmstudio_finetune_integration import FineTuneAssistant

async def intelligent_optimization():
    # Get initial suggestions from AI
    assistant = FineTuneAssistant()
    await assistant.initialize()
    
    suggestions = await assistant.suggest_hyperparameters(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_size=1000,
        task_type="text_generation"
    )
    
    # Use suggestions to narrow Optuna search space
    search_space = {
        "learning_rate": (
            suggestions["learning_rate"] * 0.5,
            suggestions["learning_rate"] * 2.0
        ),
        "batch_size": [
            suggestions["batch_size"] // 2,
            suggestions["batch_size"],
            suggestions["batch_size"] * 2
        ],
        # ...
    }
    
    # Run Optuna with narrowed space
    best_params = optuna_optimize(
        base_config="config.yaml",
        search_space=search_space,
        n_trials=20
    )

asyncio.run(intelligent_optimization())
```

### 3. Continuous Monitoring

Set up continuous monitoring with AI diagnostics:

```python
import time
from pathlib import Path
from lmstudio_finetune_integration import FineTuneAssistant

async def monitor_training(output_dir: str, check_interval: int = 300):
    """Monitor training and provide periodic diagnostics"""
    assistant = FineTuneAssistant()
    await assistant.initialize()
    
    state_file = Path(output_dir) / "trainer_state.json"
    
    while True:
        if state_file.exists():
            import json
            with open(state_file) as f:
                state = json.load(f)
            
            # Diagnose every N minutes
            diagnosis = await assistant.diagnose_training_issues(
                log_history=state.get("log_history", []),
                config={}
            )
            
            if diagnosis.get("needs_attention"):
                print(f"⚠️  Training issue detected: {diagnosis}")
                # Send alert, adjust parameters, etc.
            
        await asyncio.sleep(check_interval)

# Run in background
asyncio.create_task(monitor_training("outputs/run-20250118"))
```

## MCP Server Integration

The LM Studio integration includes 10 MCP (Model Context Protocol) servers for extended functionality:

- **filesystem**: File and directory operations
- **git**: Version control operations
- **memory**: Conversation memory and context
- **sequential-thinking**: Multi-step reasoning
- **context7**: Documentation retrieval
- **gitkraken**: Advanced git features
- **microsoft-doc**: Microsoft documentation
- **pylance**: Python language features
- **markitdown**: Markdown conversion
- **aitk**: AI toolkit features

These servers are automatically loaded and available to the AI assistant for enhanced capabilities.

## Configuration

### LM Studio Server

Configure in `~/.lmstudio/config/mcp.json`:

```json
{
  "lmstudio": {
    "url": "http://localhost:1234/v1",
    "model": "local-model",
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

### Audit Logging

All operations are logged to `~/.lmstudio/audit-logs/` for tracking and debugging.

## Troubleshooting

### LM Studio Not Running

```
Error: Failed to connect to LM Studio
```

**Solution**: Start LM Studio and ensure the local server is running on port 1234.

### Plugin Not Found

```
Error: Plugin 'finetune_manager' not found
```

**Solution**: Ensure plugins are in the `plugins/` directory with correct structure.

### Import Errors

```
ImportError: LM Studio integration not available
```

**Solution**: Copy `enhanced-cli.py` to workspace as `lmstudio_integration.py`

## Best Practices

1. **Start with Analysis**: Always analyze your dataset before training
2. **Use Suggestions**: Apply AI-recommended hyperparameters as starting points
3. **Monitor Continuously**: Set up monitoring for long training runs
4. **Diagnose Early**: Check diagnostics if loss doesn't improve
5. **Compare Checkpoints**: Use evaluation plugin to find best model
6. **Document Everything**: Generate reports for each experiment

## Advanced Usage

### Custom LLM Prompts

Customize the assistant's behavior by modifying prompts in `lmstudio_finetune_integration.py`:

```python
# Add domain-specific context
system_prompt = """You are an expert ML engineer specializing in 
fine-tuning language models for medical text analysis. 
Provide guidance specific to medical domain challenges..."""

response = await self.llm.chat([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_question}
])
```

### Batch Processing

Process multiple datasets or experiments:

```python
async def batch_analyze(datasets: List[str]):
    assistant = FineTuneAssistant()
    await assistant.initialize()
    
    results = []
    for dataset in datasets:
        analysis = await assistant.analyze_dataset(dataset)
        results.append({
            "dataset": dataset,
            "analysis": analysis
        })
    
    return results
```

## Examples

See the `/examples/` directory for complete examples:

- `example_dataset_analysis.py` - Dataset analysis workflow
- `example_intelligent_training.py` - AI-assisted training
- `example_automated_optimization.py` - Combined AI + Optuna optimization
- `example_continuous_monitoring.py` - Real-time monitoring setup

## Support

For issues or questions:
1. Check the audit logs in `~/.lmstudio/audit-logs/`
2. Verify LM Studio is running and accessible
3. Ensure all dependencies are installed
4. Review plugin execution logs

## License

Same as the main fine-tuning framework.
