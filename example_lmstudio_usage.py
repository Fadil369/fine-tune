"""
Example: AI-Assisted Fine-Tuning with LM Studio
Demonstrates the complete workflow with intelligent assistance
"""

import asyncio
import json
from pathlib import Path
from lmstudio_finetune_integration import FineTuneAssistant


async def example_workflow():
    """Complete example of AI-assisted fine-tuning workflow"""
    
    print("=" * 80)
    print("AI-Assisted Fine-Tuning Example")
    print("=" * 80)
    
    # Initialize the assistant
    print("\n1. Initializing AI Assistant...")
    try:
        assistant = FineTuneAssistant()
        await assistant.initialize()
        print("✓ Assistant initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nMake sure LM Studio is running: http://localhost:1234")
        return
    
    # Step 1: Analyze dataset
    print("\n2. Analyzing Dataset...")
    dataset_path = "data/train.jsonl"  # Update with your dataset path
    
    if Path(dataset_path).exists():
        analysis = await assistant.analyze_dataset(dataset_path)
        print(f"\nDataset Analysis:")
        print(json.dumps(analysis, indent=2))
        
        # Check for critical issues
        if "error" in analysis:
            print(f"\n✗ Analysis failed: {analysis['error']}")
            return
    else:
        print(f"⚠️  Dataset not found: {dataset_path}")
        print("Skipping dataset analysis...")
    
    # Step 2: Get hyperparameter suggestions
    print("\n3. Getting Hyperparameter Suggestions...")
    suggestions = await assistant.suggest_hyperparameters(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_size=1000,
        task_type="text_generation"
    )
    
    print(f"\nHyperparameter Suggestions:")
    print(json.dumps(suggestions, indent=2))
    
    # Step 3: Interactive Q&A (optional)
    print("\n4. Interactive Q&A Example...")
    questions = [
        "What are the most important hyperparameters for fine-tuning?",
        "How can I prevent overfitting on a small dataset?",
        "What's the difference between LoRA and full fine-tuning?"
    ]
    
    for i, question in enumerate(questions[:2], 1):  # Ask first 2 questions
        print(f"\nQ{i}: {question}")
        response = await assistant.llm.chat([
            {
                "role": "system",
                "content": "You are an expert ML engineer. Provide concise, practical answers."
            },
            {
                "role": "user",
                "content": question
            }
        ], temperature=0.3, max_tokens=200)
        print(f"A{i}: {response}")
    
    # Step 4: Simulate training and diagnose
    print("\n5. Training Diagnostics Example...")
    
    # Example log history (replace with actual training logs)
    example_log_history = [
        {"step": 0, "loss": 3.5, "epoch": 0.0},
        {"step": 100, "loss": 2.8, "epoch": 0.5},
        {"step": 200, "loss": 2.5, "epoch": 1.0},
        {"step": 300, "loss": 2.4, "epoch": 1.5},
        {"step": 400, "loss": 2.35, "epoch": 2.0}
    ]
    
    example_config = {
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "lora_r": 8,
        "lora_alpha": 16
    }
    
    diagnosis = await assistant.diagnose_training_issues(
        log_history=example_log_history,
        config=example_config
    )
    
    print(f"\nTraining Diagnosis:")
    print(json.dumps(diagnosis, indent=2))
    
    # Step 5: Generate report (if training output exists)
    print("\n6. Report Generation Example...")
    
    # Check for actual training output
    output_dirs = list(Path("outputs").glob("*")) if Path("outputs").exists() else []
    
    if output_dirs:
        latest_output = sorted(output_dirs, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Generating report for: {latest_output}")
        
        # Load metrics if available
        state_file = latest_output / "trainer_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            metrics = state.get("log_history", [{}])[-1]
            
            report = await assistant.generate_training_report(
                str(latest_output),
                metrics
            )
            
            print(f"\n{report}")
            
            # Save report to file
            report_file = latest_output / "ai_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\n✓ Report saved to: {report_file}")
        else:
            print("⚠️  No trainer state found, skipping report generation")
    else:
        print("⚠️  No training outputs found, skipping report generation")
    
    # Summary
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use the suggestions to update your config.yaml")
    print("2. Run training: python finetune.py --config config.yaml")
    print("3. Monitor with: python lmstudio_finetune_integration.py --mode diagnose")
    print("4. Compare results with plugins/model_evaluator.py")
    print("5. Deploy with deployment_utils.py")
    

async def example_plugin_usage():
    """Example of using plugins directly"""
    
    print("\n" + "=" * 80)
    print("Plugin Usage Examples")
    print("=" * 80)
    
    # Fine-Tune Manager Plugin
    print("\n1. Fine-Tune Manager Plugin")
    from plugins.finetune_manager import get_plugin as get_manager
    
    manager = get_manager()
    
    # List all training runs
    runs = manager.execute("list")
    print(f"Training runs: {json.dumps(runs, indent=2)}")
    
    # Get status of latest run
    if runs["runs"]:
        latest = runs["runs"][-1]
        status = manager.execute("status", output_dir=latest["path"])
        print(f"Latest run status: {json.dumps(status, indent=2)}")
    
    # Hyperparameter Assistant Plugin
    print("\n2. Hyperparameter Assistant Plugin")
    from plugins.hyperparameter_assistant import get_plugin as get_assistant
    
    assistant = get_assistant()
    
    # Get suggestions
    suggestions = assistant.execute("suggest",
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_size=1000,
        task_type="text_generation",
        gpu_memory_gb=24
    )
    print(f"Suggestions: {json.dumps(suggestions, indent=2)}")
    
    # Model Evaluator Plugin
    print("\n3. Model Evaluator Plugin")
    from plugins.model_evaluator import get_plugin as get_evaluator
    
    evaluator = get_evaluator()
    
    # List checkpoints from outputs directory
    if Path("outputs").exists():
        output_dirs = list(Path("outputs").iterdir())
        if output_dirs:
            latest_output = output_dirs[0]
            checkpoints = evaluator.execute("list_checkpoints",
                output_dir=str(latest_output)
            )
            print(f"Checkpoints: {json.dumps(checkpoints, indent=2)}")


async def main():
    """Run all examples"""
    
    # Main workflow
    await example_workflow()
    
    # Plugin examples
    await example_plugin_usage()


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║            AI-Assisted Fine-Tuning Framework - Example Script              ║
║                                                                            ║
║  This example demonstrates the complete workflow with LM Studio           ║
║  integration, including dataset analysis, hyperparameter suggestions,     ║
║  training diagnostics, and report generation.                             ║
║                                                                            ║
║  Prerequisites:                                                            ║
║  1. LM Studio running on http://localhost:1234                            ║
║  2. A model loaded in LM Studio (e.g., Mistral 7B)                        ║
║  3. Python dependencies installed: pip install requests rich aiohttp      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure LM Studio is running: http://localhost:1234")
        print("2. Check that lmstudio_integration.py exists in the workspace")
        print("3. Verify all dependencies are installed")
