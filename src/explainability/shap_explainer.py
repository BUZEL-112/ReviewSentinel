"""
SHAP Explainability Module

Provides the SentimentExplainer class to generate SHAP values for transformer predictions.
"""
import torch
import numpy as np
import shap
from src.utils.logger import logger
from src.utils.exception import CustomException

class SentimentExplainer:
    def __init__(self, model, tokenizer, device, max_evals=500, label_map=None, top_k_tokens=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_evals = max_evals
        self.label_map = label_map if label_map is not None else {0: "negative", 1: "neutral", 2: "positive"}
        self.top_k_tokens = top_k_tokens
        
        self.predict_fn = self._build_predict_fn()
        self.masker = shap.maskers.Text(r"\W+")
        self.explainer = None

    def _build_predict_fn(self):
        def predict_fn(texts):
            encodings = self.tokenizer(
                list(texts),
                max_length=self.tokenizer.model_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                logits = self.model(**encodings).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        return predict_fn

    def _aggregate_subword_tokens(self, tokens, shap_values):
        """
        Handles merging subword tokens back to words if subword masking is used.
        Currently unused because we use shap.maskers.Text(r"\W+") which is word-level.
        """
        pass

    def explain(self, text: str, target_class: str = None) -> dict:
        try:
            if self.explainer is None:
                logger.info("Initializing SHAP explainer instance...")
                self.explainer = shap.Explainer(self.predict_fn, self.masker)
                
            shap_values = self.explainer([text], max_evals=self.max_evals)
            
            vals = shap_values.values[0]
            tokens = shap_values.data[0]
            
            if target_class is None:
                probs = self.predict_fn([text])[0]
                class_idx = np.argmax(probs)
                target_class = self.label_map[class_idx]
            else:
                inv_map = {v: k for k, v in self.label_map.items()}
                class_idx = inv_map.get(target_class, 0)
                
            class_vals = vals[:, class_idx]
            baseline_prob = shap_values.base_values[0][class_idx]
            
            token_scores = []
            for t, v in zip(tokens, class_vals):
                t_str = str(t).strip()
                if not t_str or t_str in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    continue
                direction = "toward" if v > 0 else "against"
                token_scores.append({
                    "token": t_str,
                    "shap_value": round(float(v), 4),
                    "direction": direction
                })
                
            token_scores.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            token_scores = token_scores[:self.top_k_tokens]
            
            return {
                "predicted_class": target_class,
                "target_class_explained": target_class,
                "baseline_probability": round(float(baseline_prob), 4),
                "tokens": token_scores
            }
        except Exception as e:
            logger.error(f"SHAP explainer failed: {e}")
            raise CustomException(e)
