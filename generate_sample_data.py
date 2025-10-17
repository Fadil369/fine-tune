#!/usr/bin/env python3
"""
Generate sample datasets for testing the fine-tuning framework
"""

import json
import argparse
from pathlib import Path


def generate_text_generation_data(num_samples=100, output_file="sample_text_gen.json"):
    """Generate sample data for text generation tasks"""
    samples = []
    
    topics = [
        ("AI", "Artificial Intelligence is the simulation of human intelligence by machines."),
        ("ML", "Machine Learning is a subset of AI that enables systems to learn from data."),
        ("DL", "Deep Learning uses neural networks with multiple layers to learn complex patterns."),
        ("NLP", "Natural Language Processing helps computers understand human language."),
        ("CV", "Computer Vision enables machines to interpret visual information."),
    ]
    
    templates = [
        "The concept of {topic} is important because {description}",
        "{description} This is what {topic} is all about.",
        "When discussing {topic}, we should note that {description}",
        "{topic} refers to the following: {description}",
        "Understanding {topic} is crucial. {description}",
    ]
    
    for i in range(num_samples):
        topic, description = topics[i % len(topics)]
        template = templates[i % len(templates)]
        text = template.format(topic=topic, description=description)
        samples.append({"text": text})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} text generation samples → {output_file}")


def generate_instruction_data(num_samples=50, output_file="sample_instructions.json"):
    """Generate sample instruction-following data"""
    samples = []
    
    qa_pairs = [
        ("What is Python?", "Python is a high-level, interpreted programming language known for its simplicity and readability."),
        ("Explain machine learning", "Machine learning is a method of data analysis that automates analytical model building."),
        ("How does a neural network work?", "Neural networks process data through interconnected layers of nodes that learn patterns."),
        ("What is an API?", "An API (Application Programming Interface) is a set of protocols for building software applications."),
        ("Define cloud computing", "Cloud computing is the delivery of computing services over the internet."),
        ("What is version control?", "Version control is a system that records changes to files over time."),
        ("Explain REST API", "REST API is an architectural style for designing networked applications using HTTP requests."),
        ("What is Docker?", "Docker is a platform for developing, shipping, and running applications in containers."),
        ("Define DevOps", "DevOps is a set of practices combining software development and IT operations."),
        ("What is Git?", "Git is a distributed version control system for tracking changes in source code."),
    ]
    
    for i in range(num_samples):
        question, answer = qa_pairs[i % len(qa_pairs)]
        
        # Format as instruction-following
        text = f"### Instruction:\n{question}\n\n### Response:\n{answer}"
        samples.append({"text": text})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} instruction samples → {output_file}")


def generate_classification_data(num_samples=200, output_file="sample_classification.json"):
    """Generate sample classification data"""
    samples = []
    
    # Sentiment classification examples
    positive_texts = [
        "This product is amazing! Highly recommended.",
        "Excellent quality and fast shipping.",
        "Best purchase I've made this year!",
        "Absolutely love it, exceeds expectations.",
        "Great value for money, very satisfied.",
        "Outstanding performance and reliability.",
        "Couldn't be happier with this product.",
        "Five stars! Will definitely buy again.",
        "Impressive quality and attention to detail.",
        "Fantastic! Worth every penny.",
    ]
    
    negative_texts = [
        "Terrible product, complete waste of money.",
        "Very disappointed with the quality.",
        "Do not recommend, poor performance.",
        "Worst purchase ever, broke immediately.",
        "Not worth the price, very cheap materials.",
        "Customer service was horrible.",
        "Arrived damaged and incomplete.",
        "Does not work as advertised.",
        "Extremely unsatisfied with this product.",
        "Save your money, buy something else.",
    ]
    
    # Generate balanced dataset
    for i in range(num_samples):
        if i % 2 == 0:
            text = positive_texts[i % len(positive_texts)]
            label = 1  # Positive
        else:
            text = negative_texts[i % len(negative_texts)]
            label = 0  # Negative
        
        samples.append({"text": text, "label": label})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} classification samples → {output_file}")


def generate_conversation_data(num_samples=50, output_file="sample_conversations.json"):
    """Generate sample conversation data"""
    samples = []
    
    conversations = [
        ("Hello! How can I help you today?", "Hi! I need help with setting up my account."),
        ("What programming language should I learn first?", "I recommend starting with Python for its simplicity and versatility."),
        ("How do I debug my code?", "Use print statements, debuggers, and check error messages carefully."),
        ("What is the best way to learn coding?", "Practice regularly, build projects, and learn from others' code."),
        ("Can you explain what a function is?", "A function is a reusable block of code that performs a specific task."),
        ("How do I improve my code quality?", "Follow best practices, write tests, and review code regularly."),
        ("What is the difference between AI and ML?", "AI is the broader concept, while ML is a specific approach to achieving AI."),
        ("How can I speed up my website?", "Optimize images, use caching, minify code, and use a CDN."),
        ("What database should I use?", "It depends on your needs - SQL for structured data, NoSQL for flexibility."),
        ("How do I start a tech career?", "Build a portfolio, contribute to open source, and network with professionals."),
    ]
    
    for i in range(num_samples):
        user_msg, assistant_msg = conversations[i % len(conversations)]
        
        text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        samples.append({"text": text})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} conversation samples → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample datasets")
    parser.add_argument(
        "--type",
        type=str,
        choices=["text_gen", "instruction", "classification", "conversation", "all"],
        default="all",
        help="Type of dataset to generate"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_data",
        help="Output directory for generated files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating sample datasets in {output_dir}/")
    print("="*60)
    
    if args.type in ["text_gen", "all"]:
        generate_text_generation_data(
            args.num_samples,
            output_dir / "text_generation.json"
        )
    
    if args.type in ["instruction", "all"]:
        generate_instruction_data(
            args.num_samples,
            output_dir / "instruction.json"
        )
    
    if args.type in ["classification", "all"]:
        generate_classification_data(
            args.num_samples * 2,  # More samples for balanced dataset
            output_dir / "classification.json"
        )
    
    if args.type in ["conversation", "all"]:
        generate_conversation_data(
            args.num_samples,
            output_dir / "conversation.json"
        )
    
    print("="*60)
    print("\nSample datasets generated successfully!")
    print("\nNext steps:")
    print(f"1. Split data: python prepare_data.py --input {output_dir}/instruction.json --task split")
    print("2. Edit config.yaml to point to your training data")
    print("3. Run: python finetune.py --config config.yaml")


if __name__ == "__main__":
    main()
