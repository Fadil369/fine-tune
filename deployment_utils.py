"""
Production Deployment Utilities
FastAPI server, Gradio interface, and Docker support
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== FastAPI Server =====

FASTAPI_CODE = '''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="Fine-Tuned Model API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None


class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1


class GenerationResponse(BaseModel):
    generated_text: List[str]
    prompt: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer
    
    model_path = os.getenv("MODEL_PATH", "./output")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Model loaded on {device}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        device = next(model.parameters()).device
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                num_return_sequences=request.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return GenerationResponse(
            generated_text=generated_texts,
            prompt=request.prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(next(model.parameters()).device)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


# ===== Gradio Interface =====

GRADIO_CODE = '''
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "./output")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
print(f"Model loaded on {device}")


def generate_text(
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """Generate text from prompt"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here...",
            lines=3
        ),
        gr.Slider(
            minimum=50,
            maximum=1024,
            value=256,
            step=1,
            label="Max Length"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top P"
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=50,
            step=1,
            label="Top K"
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Fine-Tuned Model Interface",
    description="Generate text using your fine-tuned model",
    examples=[
        ["What is machine learning?", 256, 0.7, 0.9, 50],
        ["Explain neural networks", 256, 0.7, 0.9, 50],
        ["How does AI work?", 256, 0.7, 0.9, 50],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
'''


# ===== Dockerfile =====

DOCKERFILE = '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY output/ ./output/
COPY *.py ./

# Set environment variable
ENV MODEL_PATH=/app/output
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "api_server.py"]
'''


# ===== Docker Compose =====

DOCKER_COMPOSE = '''
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/output
    volumes:
      - ./output:/app/output:ro
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  gradio-ui:
    build: .
    command: python gradio_app.py
    ports:
      - "7860:7860"
    environment:
      - MODEL_PATH=/app/output
    volumes:
      - ./output:/app/output:ro
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''


# ===== Requirements for deployment =====

DEPLOYMENT_REQUIREMENTS = '''
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
gradio==4.7.1
'''


def create_deployment_files(output_dir: str = "./deployment"):
    """
    Create all deployment files
    
    Args:
        output_dir: Directory to save deployment files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Creating deployment files in {output_dir}")
    
    # FastAPI server
    with open(output_path / "api_server.py", 'w') as f:
        f.write(FASTAPI_CODE)
    logger.info("✓ Created api_server.py")
    
    # Gradio interface
    with open(output_path / "gradio_app.py", 'w') as f:
        f.write(GRADIO_CODE)
    logger.info("✓ Created gradio_app.py")
    
    # Dockerfile
    with open(output_path / "Dockerfile", 'w') as f:
        f.write(DOCKERFILE)
    logger.info("✓ Created Dockerfile")
    
    # Docker Compose
    with open(output_path / "docker-compose.yml", 'w') as f:
        f.write(DOCKER_COMPOSE)
    logger.info("✓ Created docker-compose.yml")
    
    # Deployment requirements
    with open(output_path / "requirements-deploy.txt", 'w') as f:
        f.write(DEPLOYMENT_REQUIREMENTS)
    logger.info("✓ Created requirements-deploy.txt")
    
    # README
    readme = """
# Deployment Guide

## FastAPI Server

### Local Development
```bash
pip install -r requirements-deploy.txt
export MODEL_PATH=../output
python api_server.py
```

Visit: http://localhost:8000/docs

### Docker
```bash
docker build -t model-api .
docker run -p 8000:8000 -v $(pwd)/output:/app/output model-api
```

## Gradio Interface

### Local
```bash
export MODEL_PATH=../output
python gradio_app.py
```

Visit: http://localhost:7860

### Docker Compose
```bash
docker-compose up -d
```

- API: http://localhost:8000
- UI: http://localhost:7860

## Cloud Deployment

### AWS EC2
1. Launch EC2 instance with GPU
2. Install Docker and NVIDIA Container Toolkit
3. Clone repository and build image
4. Run container

### Azure Container Instances
```bash
az container create \\
  --resource-group myResourceGroup \\
  --name model-api \\
  --image myregistry.azurecr.io/model-api:latest \\
  --gpu-count 1 \\
  --gpu-sku K80 \\
  --ports 8000
```

### Google Cloud Run
```bash
gcloud run deploy model-api \\
  --image gcr.io/myproject/model-api \\
  --platform managed \\
  --region us-central1 \\
  --memory 4Gi \\
  --gpu 1
```

## API Usage

### cURL
```bash
curl -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "What is AI?", "max_length": 256}'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "What is AI?", "max_length": 256}
)
print(response.json())
```

## Monitoring

Add health checks, logging, and metrics as needed.
"""
    
    with open(output_path / "README.md", 'w') as f:
        f.write(readme)
    logger.info("✓ Created README.md")
    
    logger.info(f"\n{'='*80}")
    logger.info("Deployment files created successfully!")
    logger.info(f"Location: {output_dir}")
    logger.info(f"{'='*80}\n")
    
    logger.info("Next steps:")
    logger.info("1. cd deployment")
    logger.info("2. pip install -r requirements-deploy.txt")
    logger.info("3. python api_server.py  # or python gradio_app.py")
    logger.info("4. For Docker: docker-compose up -d")


def main():
    """CLI for deployment utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create deployment files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./deployment",
        help="Directory to save deployment files"
    )
    
    args = parser.parse_args()
    create_deployment_files(args.output_dir)


if __name__ == "__main__":
    main()
