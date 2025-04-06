# embedding.py
import numpy as np
import torch
from transformers import DistilBertModel, DistilBertTokenizer

class EmbeddingModel:
    """
    Modèle pour générer des embeddings neuronaux en utilisant DistilBERT.
    """
    def __init__(self, embedding_size=128):
        """
        Initialise le modèle d'embedding.
        
        Args:
            embedding_size: Dimension des embeddings
        """
        self.embedding_size = embedding_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    def embed_text(self, text):
        """
        Génère l'embedding pour un texte.
        
        Args:
            text: Texte à encoder
            
        Returns:
            Vecteur d'embedding
        """
        # Tokenise le texte
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Génère l'embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Utilise le token [CLS] comme représentation du modèle
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        # Réduit la dimensionnalité si nécessaire
        if embedding.shape[0] > self.embedding_size:
            # Réduction de dimensionnalité simple par moyenne
            chunks = np.array_split(embedding, self.embedding_size)
            embedding = np.array([chunk.mean() for chunk in chunks])
        
        return embedding