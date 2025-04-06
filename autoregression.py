# autoregression.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ModelTransformation:
    """
    Représente une transformation de modèle avec modèles source et cible.
    """
    def __init__(self, source_model, target_model, source_metamodel, target_metamodel):
        """
        Initialise une transformation de modèle.
        """
        self.source_model = source_model
        self.target_model = target_model
        self.source_metamodel = source_metamodel
        self.target_metamodel = target_metamodel
        self.source_token_pairs = []
        self.target_token_pairs = []
        self.similarity_score = None
        self.is_cross_metamodel = source_metamodel != target_metamodel

    def to_features(self, embedding_model=None):
        """
        Convertit la transformation en vecteur de caractéristiques pour l'auto-régression.
        """
        features = []
        
        # Caractéristiques statistiques de base
        source_count = len(self.source_token_pairs)
        target_count = len(self.target_token_pairs)
        ratio = target_count / source_count if source_count > 0 else 0
        
        features.extend([source_count, target_count, ratio])
        
        # Ajoute le score de similarité si disponible
        if self.similarity_score is not None:
            features.append(self.similarity_score)
        
        # Ajoute les embeddings si le modèle est disponible
        if embedding_model is not None:
            # Génère des embeddings pour les modèles source et cible
            source_emb = embedding_model.embed_text(self.source_model)
            target_emb = embedding_model.embed_text(self.target_model)
            
            # Ajoute aux caractéristiques
            features.extend(source_emb)
            features.extend(target_emb)
        
        return features

class TransformationHistoryDataset(Dataset):
    """
    Dataset pour entraîner le modèle d'auto-régression avec l'historique des transformations.
    """
    def __init__(self, transformations, window_size=3, embedding_model=None):
        """
        Initialise le dataset.
        """
        self.transformations = transformations
        self.window_size = window_size
        self.embedding_model = embedding_model
        self.seq_data = []
        
        # Crée des séquences
        for i in range(len(transformations) - window_size):
            # Fenêtre historique
            window = transformations[i:i+window_size]
            
            # Transformation actuelle
            current = transformations[i+window_size]
            
            # Crée des vecteurs de caractéristiques à partir de la fenêtre
            window_features = []
            for trans in window:
                # Extrait les caractéristiques de la transformation
                trans_features = trans.to_features(embedding_model)
                window_features.extend(trans_features)
            
            # Crée des vecteurs de caractéristiques pour la transformation actuelle
            current_features = current.to_features(embedding_model)
            
            # La cible est le score de similarité de la transformation actuelle
            target = current.similarity_score
            
            # Les caractéristiques X combinent la fenêtre et la transformation actuelle
            X = window_features + current_features
            y = target
            
            self.seq_data.append((X, y))
    
    def __len__(self):
        return len(self.seq_data)
    
    def __getitem__(self, idx):
        X, y = self.seq_data[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class AutoRegressionModel(nn.Module):
    """
    Modèle d'auto-régression pour l'évaluation de la préservation sémantique.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        """
        Initialise le modèle.
        """
        super(AutoRegressionModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,  # Réduire le nombre de couches
            batch_first=True,
            dropout=0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Assure que la sortie est entre 0 et 1
        )
    
    def forward(self, x):
        """
        Passe en avant.
        """
        # Traite par LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Utilise l'état caché final
        out = self.fc(h_n[-1])
        
        return out