"""
FastAPI server for Multimodal Vision AI
Endpoints: /caption, /vqa, /batch
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import List
import torch
from model import MultimodalVisionModel, VQAModel

app = FastAPI(title="Multimodal Vision AI API")

# Load models
caption_model = MultimodalVisionModel()
vqa_model = VQAModel()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model = caption_model.to(device)
vqa_model = vqa_model.to(device)

@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Generate image caption
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        caption: Generated caption text
        confidence: Model confidence score
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate caption
        caption = caption_model.generate_caption(image)
        
        return JSONResponse({
            "caption": caption,
            "status": "success"
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "failed"
        }, status_code=500)

@app.post("/vqa")
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Answer questions about an image
    
    Args:
        file: Image file
        question: Question text
    
    Returns:
        answer: Generated answer
        question: Original question
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Answer question
        answer = vqa_model.answer_question(image, question)
        
        return JSONResponse({
            "question": question,
            "answer": answer,
            "status": "success"
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "failed"
        }, status_code=500)

@app.post("/batch/caption")
async def batch_caption(files: List[UploadFile] = File(...)):
    """
    Generate captions for multiple images
    
    Args:
        files: List of image files
    
    Returns:
        results: List of captions
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            caption = caption_model.generate_caption(image)
            
            results.append({
                "filename": file.filename,
                "caption": caption,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return JSONResponse({"results": results})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": str(device)}

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "vision_model": "CLIP ViT-B/32",
        "text_model": "GPT-2",
        "embedding_dim": 768,
        "max_caption_length": 50,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
