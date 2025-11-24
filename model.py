"""
Multimodal Vision AI - Image Captioning with CLIP + GPT-2
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import torchvision.transforms as transforms

class MultimodalVisionModel(nn.Module):
    """
    Vision-Language model combining CLIP vision encoder with GPT-2 decoder
    """
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32", 
                 text_model_name="gpt2"):
        super().__init__()
        
        # CLIP Vision Encoder (512-dim embeddings)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        
        # GPT-2 Decoder
        self.text_decoder = GPT2LMHeadModel.from_pretrained(text_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)
        
        # Projection layer: CLIP (512) -> GPT-2 (768)
        self.vision_projection = nn.Linear(512, 768)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(768, 8, batch_first=True)
        
    def encode_image(self, images):
        """Extract visual features using CLIP"""
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=images)
            image_features = outputs.last_hidden_state  # (batch, 50, 512)
        
        # Project to GPT-2 dimension
        image_features = self.vision_projection(image_features)  # (batch, 50, 768)
        return image_features
    
    def forward(self, images, captions=None):
        """
        Forward pass for training
        Args:
            images: (batch, 3, 224, 224)
            captions: (batch, seq_len) - tokenized captions
        """
        # Encode images
        image_features = self.encode_image(images)
        
        if captions is not None:
            # Get text embeddings
            text_embeds = self.text_decoder.transformer.wte(captions)
            
            # Cross-attention: text attends to image
            attended_text, _ = self.cross_attention(
                text_embeds, image_features, image_features
            )
            
            # Generate through GPT-2
            outputs = self.text_decoder(inputs_embeds=attended_text, labels=captions)
            return outputs
        else:
            # Inference mode
            return image_features
    
    @torch.no_grad()
    def generate_caption(self, image, max_length=50, beam_size=5, temperature=0.9):
        """
        Generate caption for a single image
        Args:
            image: PIL Image or tensor (3, 224, 224)
            max_length: maximum caption length
            beam_size: beam search width
        Returns:
            caption: generated text
        """
        self.eval()
        
        # Preprocess image
        if isinstance(image, Image.Image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0)
        
        # Encode image
        image_features = self.encode_image(image)
        
        # Start token
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]])
        
        # Generate with beam search
        generated = self.text_decoder.generate(
            inputs_embeds=image_features,
            max_length=max_length,
            num_beams=beam_size,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        caption = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return caption

class VQAModel(MultimodalVisionModel):
    """Visual Question Answering extension"""
    
    def answer_question(self, image, question):
        """
        Answer a question about an image
        Args:
            image: PIL Image
            question: text question
        Returns:
            answer: generated answer
        """
        # Encode image
        image_features = self.encode_image(self.preprocess_image(image))
        
        # Encode question
        question_ids = self.tokenizer.encode(question, return_tensors="pt")
        question_embeds = self.text_decoder.transformer.wte(question_ids)
        
        # Cross-attention
        attended_question, _ = self.cross_attention(
            question_embeds, image_features, image_features
        )
        
        # Generate answer
        output = self.text_decoder.generate(
            inputs_embeds=attended_question,
            max_length=30,
            num_beams=3
        )
        
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer

def load_model(checkpoint_path=None):
    """Load pre-trained model"""
    model = MultimodalVisionModel()
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

if __name__ == "__main__":
    # Example usage
    model = load_model()
    
    # Load image
    image = Image.open("example.jpg")
    
    # Generate caption
    caption = model.generate_caption(image)
    print(f"Caption: {caption}")
