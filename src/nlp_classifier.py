from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

class DistressClassifier:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model using safetensors (avoids PyTorch pickle security block)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,  # ← KEY FIX
            local_files_only=False
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.danger_labels = {"anger", "fear", "sadness"}
        self.label_map = self.model.config.id2label
        
        logging.info(f"✅ Emotion classifier loaded on {device} using '{model_name}' (safetensors)")

    def is_distress(self, text: str, threshold=0.6) -> bool:
        if not text.strip():
            return False
            
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            for idx, prob in enumerate(probs):
                label = self.label_map[idx].lower()
                if label in self.danger_labels and prob >= threshold:
                    logging.warning(f"🚨 DISTRESS DETECTED: '{label}' ({prob:.2f}) in '{text}'")
                    return True
                    
            logging.debug(f"😊 Safe: {dict(zip([v.lower() for v in self.label_map.values()], [f'{p:.2f}' for p in probs]))}")
            return False
            
        except Exception as e:
            logging.error(f"NLP classification failed: {e}")
            return False