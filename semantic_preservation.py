# semantic_preservation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import logging

# Import des modules personnalisés
from token_pair import TokenPair, TokenPairExtractor, TokenPairSimilarityCalculator
from embedding import EmbeddingModel
from autoregression import ModelTransformation, TransformationHistoryDataset, AutoRegressionModel

# Configuration de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SemanticPreservation")

class SemanticPreservationFramework:
    """
    Framework pour la mesure de la préservation sémantique dans les transformations de modèles.
    """
    def __init__(self, embedding_size=128, window_size=2, alpha=0.5, beta=0.7, use_embeddings=True):
        """
        Initialise le framework.
        """
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.alpha = alpha  # Poids de l'évaluation forward
        self.beta = beta    # Poids de la similarité des paires token dans l'évaluation backward
        self.use_embeddings = use_embeddings
        
        # Initialise le modèle d'embedding si nécessaire
        self.embedding_model = None
        if use_embeddings:
            logger.info(f"Initialisation du modèle d'embedding avec taille {embedding_size}")
            self.embedding_model = EmbeddingModel(embedding_size)
        
        # Initialisation de l'extracteur de paires token et du calculateur de similarité
        self.extractor = TokenPairExtractor()
        self.similarity_calculator = TokenPairSimilarityCalculator(self.embedding_model)
        
        # Historique des transformations
        self.transformation_history = []
        
        # Modèle d'auto-régression
        self.auto_regression_model = None
    
    def extract_token_pairs(self, model_text, metamodel_info):
        """
        Extrait les paires token du texte du modèle.
        """
        return self.extractor.extract_from_text(model_text, metamodel_info)
    
    def compute_token_pair_similarity(self, source_pairs, target_pairs):
        """
        Calcule la similarité entre les paires token source et cible.
        """
        if not source_pairs or not target_pairs:
            return 0.0
        
        # Crée la matrice de similarité
        similarity_matrix = np.zeros((len(source_pairs), len(target_pairs)))
        
        for i, source_pair in enumerate(source_pairs):
            for j, target_pair in enumerate(target_pairs):
                similarity = self.similarity_calculator.calculate_similarity(source_pair, target_pair)
                similarity_matrix[i, j] = similarity
        
        # Pour chaque paire source, trouve la meilleure correspondance dans la cible
        row_maxes = np.max(similarity_matrix, axis=1)
        
        # Moyenne des meilleures correspondances
        return np.mean(row_maxes)
    
    def compute_embedding_similarity(self, source_text, target_text):
        """
        Calcule la similarité en utilisant des embeddings neuronaux.
        """
        if not self.use_embeddings or self.embedding_model is None:
            return 0.5  # Score neutre si les embeddings ne sont pas utilisés
        
        # Obtient les embeddings
        source_emb = self.embedding_model.embed_text(source_text)
        target_emb = self.embedding_model.embed_text(target_text)
        
        # Calcule la similarité cosinus
        similarity = np.dot(source_emb, target_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(target_emb))
        
        # Normalise à [0, 1]
        return (similarity + 1) / 2
    
    def compute_forward_assessment(self, target_text, rules):
        """
        Calcule l'évaluation forward (correction structurelle).
        """
        if not rules:
            return 1.0  # Pas de règles à vérifier
        
        # Compte les règles satisfaites
        satisfied = 0
        for rule in rules:
            # Règle pour la transformation de Class en Table
            if "Class must be transformed to Table" in rule:
                if "Table" in target_text:
                    satisfied += 1
            # Règle pour la transformation d'Attribute en Column
            elif "Attribute must be transformed to Column" in rule:
                if "Column" in target_text:
                    satisfied += 1
            # Règle pour la préservation de la PrimaryKey
            elif "PrimaryKey constraint must be preserved" in rule:
                if "PrimaryKey" in target_text:
                    satisfied += 1
            # Autres règles
            elif rule in target_text:
                satisfied += 1
    
        return satisfied / len(rules)
    
    def compute_backward_assessment(self, source_text, target_text, source_metamodel, target_metamodel, use_embeddings=True):
        """
        Calcule l'évaluation backward (préservation sémantique).
        """
        # Extrait les paires token
        source_pairs = self.extract_token_pairs(source_text, source_metamodel)
        target_pairs = self.extract_token_pairs(target_text, target_metamodel)
        
        # Calcule la similarité des paires token
        token_pair_sim = self.compute_token_pair_similarity(source_pairs, target_pairs)
        
        if use_embeddings and self.use_embeddings:
            # Calcule la similarité d'embedding
            embedding_sim = self.compute_embedding_similarity(source_text, target_text)
            
            # Combine en utilisant beta
            return self.beta * token_pair_sim + (1 - self.beta) * embedding_sim
        else:
            return token_pair_sim
    
    def train_auto_regression_model(self, epochs=50, batch_size=8, learning_rate=0.001):
        """
        Entraîne le modèle d'auto-régression en utilisant l'historique des transformations.
        """
        if len(self.transformation_history) < self.window_size + 1:
            logger.warning(f"Pas assez d'historique de transformation pour entraîner le modèle d'auto-régression. " +
                         f"Besoin d'au moins {self.window_size + 1}, a {len(self.transformation_history)}.")
            return None
        
        logger.info(f"Entraînement du modèle d'auto-régression sur {len(self.transformation_history)} transformations " +
                   f"avec une taille de fenêtre {self.window_size}...")
        
        # Crée le dataset
        dataset = TransformationHistoryDataset(
            self.transformation_history,
            window_size=self.window_size,
            embedding_model=self.embedding_model if self.use_embeddings else None
        )
        
        # Crée le data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Détermine la taille d'entrée à partir du premier batch
        sample_batch, _ = next(iter(dataloader))
        input_size = sample_batch.shape[1]
        
        # Reformate les données pour LSTM
        seq_len = 1  # Pour simplifier, nous utilisons une longueur de séquence de 1
        
        # Initialise le modèle
        model = AutoRegressionModel(input_size=input_size//seq_len, hidden_size=128)
        
        # Fonction de perte et optimiseur
        criterion = nn.MSELoss()
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
        
        # Boucle d'entraînement
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for features, targets in dataloader:
                # Reformate les caractéristiques pour LSTM [batch_size, seq_len, features]
                features = features.reshape(features.shape[0], seq_len, -1)
                
                # Passe en avant
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets.unsqueeze(1))
                
                # Passe en arrière et optimise
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Affiche la progression
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Époque {epoch+1}/{epochs}, Perte: {avg_loss:.6f}")
        
        # Évalue le modèle
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for features, targets in dataloader:
                # Reformate les caractéristiques pour LSTM
                features = features.reshape(features.shape[0], seq_len, -1)
                
                # Passe en avant
                outputs = model(features)
                val_loss = criterion(outputs, targets.unsqueeze(1))
                val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.info(f"MSE de validation: {avg_val_loss:.6f}")
        
        # Sauvegarde le modèle
        self.auto_regression_model = model
        
        return model
    
    def predict_auto_regression(self, source_text, target_text, source_metamodel, target_metamodel):
        """
        Prédit la similarité en utilisant le modèle d'auto-régression.
        """
        if self.auto_regression_model is None:
            logger.warning("Modèle d'auto-régression non encore entraîné")
            return None
        
        if len(self.transformation_history) < self.window_size:
            logger.warning(f"Pas assez d'historique de transformation pour la prédiction. " +
                         f"Besoin d'au moins {self.window_size}, a {len(self.transformation_history)}.")
            return None
        
        # Obtient l'historique récent
        recent_history = self.transformation_history[-self.window_size:]
        
        # Extrait les caractéristiques de l'historique
        history_features = []
        for trans in recent_history:
            trans_features = trans.to_features(
                self.embedding_model if self.use_embeddings else None
            )
            history_features.extend(trans_features)
        
        # Extrait les caractéristiques de la transformation actuelle
        source_pairs = self.extract_token_pairs(source_text, source_metamodel)
        target_pairs = self.extract_token_pairs(target_text, target_metamodel)
        
        # Crée un objet de transformation temporaire
        current_trans = ModelTransformation(source_text, target_text, source_metamodel, target_metamodel)
        current_trans.source_token_pairs = source_pairs
        current_trans.target_token_pairs = target_pairs
        
        # Calcule la similarité de base pour la caractéristique
        token_pair_sim = self.compute_token_pair_similarity(source_pairs, target_pairs)
        current_trans.similarity_score = token_pair_sim
        
        # Obtient les caractéristiques
        current_features = current_trans.to_features(
            self.embedding_model if self.use_embeddings else None
        )
        
        # Combine les caractéristiques
        features = history_features + current_features
        
        # Convertit en tenseur et reformate pour LSTM
        features_tensor = torch.tensor([features], dtype=torch.float32)
        features_tensor = features_tensor.reshape(1, 1, -1)  # [batch_size, seq_len, features]
        
        # Fait la prédiction
        self.auto_regression_model.eval()
        with torch.no_grad():
            prediction = self.auto_regression_model(features_tensor)
        
        return prediction.item()
    
    def assess_transformation(self, source_text, target_text, rules, source_metamodel, target_metamodel,
                         is_cross_metamodel=True, use_auto_regression=True, use_embeddings=None, update_history=True):
        """
        Évalue une transformation de modèle.
        """
        # Si use_embeddings n'est pas spécifié, utiliser la valeur par défaut de self.use_embeddings
        if use_embeddings is None:
            use_embeddings = self.use_embeddings
            
        # Extrait les paires token
        source_pairs = self.extract_token_pairs(source_text, source_metamodel)
        target_pairs = self.extract_token_pairs(target_text, target_metamodel)
        
        # Calcule l'évaluation forward (correction structurelle)
        forward_score = self.compute_forward_assessment(target_text, rules)
        
        # Calcule l'évaluation backward (préservation sémantique)
        backward_score = self.compute_backward_assessment(
            source_text, target_text, source_metamodel, target_metamodel, 
            use_embeddings=use_embeddings
        )
        
        # Reste de la méthode...
        
        # Prédiction par auto-régression
        auto_score = None
        if use_auto_regression and self.auto_regression_model is not None:
            auto_score = self.predict_auto_regression(
                source_text, target_text, source_metamodel, target_metamodel
            )
        
        # Calcule le score de qualité global
        if use_auto_regression and auto_score is not None:
            # Combine le score backward avec auto-régression
            combined_backward = 0.5 * backward_score + 0.5 * auto_score
            quality_score = self.alpha * forward_score + (1 - self.alpha) * combined_backward
        else:
            quality_score = self.alpha * forward_score + (1 - self.alpha) * backward_score
        
        # Crée un objet de transformation pour l'historique
        transformation = ModelTransformation(source_text, target_text, source_metamodel, target_metamodel)
        transformation.source_token_pairs = source_pairs
        transformation.target_token_pairs = target_pairs
        transformation.similarity_score = quality_score
        transformation.is_cross_metamodel = is_cross_metamodel
        
        # Ajoute à l'historique si demandé
        if update_history:
            self.transformation_history.append(transformation)
        
        # Identifie les lacunes sémantiques
        semantic_gaps = self.identify_semantic_gaps(source_pairs, target_pairs)
        
        # Retourne les détails de l'évaluation
        results = {
            'forward_score': forward_score,
            'backward_score': backward_score,
            'auto_regression_score': auto_score,
            'quality_score': quality_score,
            'is_cross_metamodel': is_cross_metamodel,
            'source_token_pairs': len(source_pairs),
            'target_token_pairs': len(target_pairs),
            'semantic_gaps': semantic_gaps
        }
        
        return results
    
    def identify_semantic_gaps(self, source_pairs, target_pairs, threshold=0.5):
        """
        Identifie les lacunes sémantiques dans la transformation.
        """
        if not source_pairs or not target_pairs:
            return []
        
        gaps = []
        
        for source_pair in source_pairs:
            best_similarity = 0.0
            
            for target_pair in target_pairs:
                similarity = self.similarity_calculator.calculate_similarity(source_pair, target_pair)
                if similarity > best_similarity:
                    best_similarity = similarity
            
            if best_similarity < threshold:
                gaps.append((source_pair, best_similarity))
        
        return gaps
    
    def analyze_results(self, results_list):
        """
        Analyse les résultats d'évaluation avec focus sur les cross-metamodel.
        """
        # Sépare par type de transformation
        cross_metamodel = [r for r in results_list if r.get('is_cross_metamodel', False)]
        within_metamodel = [r for r in results_list if not r.get('is_cross_metamodel', False)]
        
        # Analyse globale
        analysis = self._compute_global_analysis(results_list)
        
        # Analyse spécifique aux cross-metamodel
        cross_analysis = self._compute_global_analysis(cross_metamodel)
        analysis['cross_metamodel_analysis'] = cross_analysis
        
        # Analyse spécifique aux within-metamodel
        within_analysis = self._compute_global_analysis(within_metamodel)
        analysis['within_metamodel_analysis'] = within_analysis
        
        # Effet différentiel des embeddings
        analysis['embedding_differential_effect'] = self._compute_embedding_differential_effect(results_list)
        
        return analysis

def _compute_global_analysis(self, results):
    """
    Calcule l'analyse globale pour un ensemble de résultats.
    """
    # Sépare par approche
    baseline = [r for r in results if r.get('approach') == 'Baseline']
    auto = [r for r in results if r.get('approach') == 'Auto-regression']
    embedding = [r for r in results if r.get('approach') == 'Embedding']
    combined = [r for r in results if r.get('approach') == 'Combined']
    
    # Calcule les scores moyens
    baseline_avg = np.mean([r['quality_score'] for r in baseline]) if baseline else None
    auto_avg = np.mean([r['quality_score'] for r in auto]) if auto else None
    embedding_avg = np.mean([r['quality_score'] for r in embedding]) if embedding else None
    combined_avg = np.mean([r['quality_score'] for r in combined]) if combined else None
    
    # Calcule les améliorations
    auto_improvement = None
    embedding_improvement = None
    combined_improvement = None
    
    if baseline_avg is not None and baseline_avg > 0:
        if auto_avg is not None:
            auto_improvement = 100 * (auto_avg - baseline_avg) / baseline_avg
        if embedding_avg is not None:
            embedding_improvement = 100 * (embedding_avg - baseline_avg) / baseline_avg
        if combined_avg is not None:
            combined_improvement = 100 * (combined_avg - baseline_avg) / baseline_avg
    
    return {
        'count': len(results),
        'baseline_avg': baseline_avg,
        'auto_avg': auto_avg,
        'embedding_avg': embedding_avg,
        'combined_avg': combined_avg,
        'auto_improvement': auto_improvement,
        'embedding_improvement': embedding_improvement,
        'combined_improvement': combined_improvement
    }

def _compute_embedding_differential_effect(self, results_list):
    """
    Calcule l'effet différentiel des embeddings entre cross et within-metamodel.
    """
    # Sépare par type de transformation
    cross = [r for r in results_list if r.get('is_cross_metamodel', False)]
    within = [r for r in results_list if not r.get('is_cross_metamodel', False)]
    
    # Calcule l'effet sur cross-metamodel
    cross_baseline = [r for r in cross if r.get('approach') == 'Baseline']
    cross_embedding = [r for r in cross if r.get('approach') == 'Embedding']
    
    cross_baseline_avg = np.mean([r['quality_score'] for r in cross_baseline]) if cross_baseline else None
    cross_embedding_avg = np.mean([r['quality_score'] for r in cross_embedding]) if cross_embedding else None
    
    cross_effect = None
    if cross_baseline_avg is not None and cross_baseline_avg > 0 and cross_embedding_avg is not None:
        cross_effect = 100 * (cross_embedding_avg - cross_baseline_avg) / cross_baseline_avg
    
    # Calcule l'effet sur within-metamodel
    within_baseline = [r for r in within if r.get('approach') == 'Baseline']
    within_embedding = [r for r in within if r.get('approach') == 'Embedding']
    
    within_baseline_avg = np.mean([r['quality_score'] for r in within_baseline]) if within_baseline else None
    within_embedding_avg = np.mean([r['quality_score'] for r in within_embedding]) if within_embedding else None
    
    within_effect = None
    if within_baseline_avg is not None and within_baseline_avg > 0 and within_embedding_avg is not None:
        within_effect = 100 * (within_embedding_avg - within_baseline_avg) / within_baseline_avg
    
    return {
        'cross_effect': cross_effect,
        'within_effect': within_effect,
        'difference': (cross_effect - within_effect) if cross_effect is not None and within_effect is not None else None
    }
    
    def visualize_results(self, results_list):
        """
        Visualise les résultats d'évaluation.
        """
        analysis = self.analyze_results(results_list)
        
        # Crée la figure
        plt.figure(figsize=(12, 10))
        
        # Graphique 1: Comparaison des approches d'évaluation
        plt.subplot(2, 2, 1)
        
        # Prépare les données
        labels = ['Baseline', 'Auto-Regression', 'Combined']
        scores = [
            analysis['baseline_avg'],
            analysis.get('auto_regression_avg', analysis['baseline_avg']),  # Utilise baseline si auto_regression non disponible
            analysis['quality_avg']
        ]
        
        # Améliorations
        improvements = [
            0,
            analysis.get('auto_improvement', 0),
            analysis.get('quality_improvement', 0)
        ]
        
        # Trace les scores de qualité
        bars = plt.bar(labels, scores, color=['lightblue', 'royalblue', 'darkblue'])
        
        # Ajoute les étiquettes de score
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('Comparaison des approches d\'évaluation')
        plt.ylabel('Score de qualité')
        plt.ylim(0.8, 1.0)  # Ajuste pour une meilleure visualisation
        
        # Ajoute les améliorations en tant que texte
        for i, improvement in enumerate(improvements):
            if improvement is not None:
                plt.annotate(f'{improvement:.2f}%',
                            xy=(i, scores[i] - 0.05),
                            ha='center', va='bottom',
                            color='red' if improvement < 0 else 'green')
        
        # Graphique 2: Cross-Metamodel vs Within-Metamodel
        plt.subplot(2, 2, 2)
        
        # Prépare les données
        intent_labels = ['Cross-Metamodel', 'Within-Metamodel']
        intent_scores = [
            analysis.get('cross_metamodel_avg', 0),
            analysis.get('within_metamodel_avg', 0)
        ]
        intent_counts = [analysis['count_cross'], analysis['count_within']]
        
        # Trace les scores d'intention
        intent_bars = plt.bar(intent_labels, intent_scores, color=['salmon', 'lightgreen'])
        
        # Ajoute les étiquettes de score
        for bar in intent_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom')
        
        # Ajoute les étiquettes de comptage
        for i, count in enumerate(intent_counts):
            plt.annotate(f'n={count}',
                        xy=(i, 0.1),
                        ha='center', va='bottom')
        
        plt.title('Comparaison des types de transformation')
        plt.ylabel('Score de qualité')
        plt.ylim(0, 1.0)
        
        # Graphique 3: Pourcentages d'amélioration
        plt.subplot(2, 2, 3)
        
        # Prépare les données
        improvement_labels = ['Auto-Regression', 'Combined']
        improvement_values = [
            analysis.get('auto_improvement', 0),
            analysis.get('quality_improvement', 0)
        ]
        
        # Trace les améliorations
        improvement_bars = plt.bar(improvement_labels, improvement_values, 
                                 color=['royalblue', 'darkblue'])
        
        # Ajoute les étiquettes d'amélioration
        for bar in improvement_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, 
                   height + 0.1 if height >= 0 else height - 0.3,
                   f'{height:.2f}%', ha='center', va='bottom')
        
        plt.title('Amélioration par rapport à la baseline')
        plt.ylabel('Amélioration (%)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Graphique 4: Résumé textuel
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = (
            f"Résumé de l'évaluation de la préservation sémantique\n\n"
            f"Évaluations totales: {len(results_list)}\n"
            f"Transformations cross-metamodel: {analysis['count_cross']}\n"
            f"Transformations within-metamodel: {analysis['count_within']}\n\n"
            f"Moyenne baseline: {analysis['baseline_avg']:.4f}\n"
            f"Moyenne auto-régression: {analysis.get('auto_regression_avg', 'N/A')}\n"
            f"Moyenne qualité combinée: {analysis['quality_avg']:.4f}\n\n"
            f"Amélioration auto-régression: {analysis.get('auto_improvement', 'N/A')}\n"
            f"Amélioration approche combinée: {analysis.get('quality_improvement', 'N/A')}\n\n"
            f"Lacunes sémantiques identifiées: {analysis['total_semantic_gaps']}"
        )
        
        plt.text(0, 1, summary_text, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('assessment_results.png')
        plt.show()
    
    def visualize_token_pair_similarity(self, source_pairs, target_pairs):
        """
        Visualise la matrice de similarité des paires token.
        """
        if not source_pairs or not target_pairs:
            logger.warning("Impossible de visualiser la similarité des paires token: Aucune paire token fournie")
            return
        
        # Calcule la matrice de similarité
        similarity_matrix = np.zeros((len(source_pairs), len(target_pairs)))
        
        for i, source_pair in enumerate(source_pairs):
            for j, target_pair in enumerate(target_pairs):
                similarity = self.similarity_calculator.calculate_similarity(source_pair, target_pair)
                similarity_matrix[i, j] = similarity
        
        # Crée les étiquettes
        source_labels = [f"{tp.element_name} ({tp.element_type})" for tp in source_pairs]
        target_labels = [f"{tp.element_name} ({tp.element_type})" for tp in target_pairs]
        
        # Crée la heat map
        plt.figure(figsize=(12, 10))
        plt.imshow(similarity_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Similarité')
        
        # Ajoute les étiquettes
        plt.xticks(np.arange(len(target_labels)), target_labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(source_labels)), source_labels)
        
        plt.title('Matrice de similarité des paires token')
        plt.tight_layout()
        plt.savefig('results/token_pair_similarity.png')
        plt.show()
    
    def visualize_semantic_gaps(self, gaps):
        """
        Visualise les lacunes sémantiques dans la transformation.
        """
        if not gaps:
            logger.info("Aucune lacune sémantique identifiée.")
            return
        
        # Trie les lacunes par similarité (croissante)
        gaps.sort(key=lambda x: x[1])
        
        # Extrait les données pour la visualisation
        pairs = [f"{g[0].element_name} ({g[0].element_type})" for g in gaps]
        similarities = [g[1] for g in gaps]
        
        # Crée le graphique à barres
        plt.figure(figsize=(10, 6))
        bars = plt.barh(pairs, similarities, color='salmon')
        
        # Ajoute les valeurs de similarité
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{similarities[i]:.2f}', va='center')
        
        plt.xlabel('Score de similarité')
        plt.title('Lacunes sémantiques dans la transformation')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Seuil')
        plt.legend()
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig('results/semantic_gaps.png')
        plt.show()

    def visualize_metamodel_difference(results):
        # Séparer par type de transformation
        cross_meta = [r for r in results if r['is_cross_metamodel']]
        within_meta = [r for r in results if not r['is_cross_metamodel']]
        
        # Calculer les moyennes par approche
        cross_baseline = np.mean([r['backward_score'] for r in cross_meta if r['approach'] == 'Baseline'])
        cross_embed = np.mean([r['backward_score'] for r in cross_meta if r['approach'] == 'Embedding'])
        within_baseline = np.mean([r['backward_score'] for r in within_meta if r['approach'] == 'Baseline'])
        within_embed = np.mean([r['backward_score'] for r in within_meta if r['approach'] == 'Baseline'])
        
        # Calculer les différences
        cross_diff = 100 * (cross_embed - cross_baseline) / cross_baseline
        within_diff = 100 * (within_embed - within_baseline) / within_baseline
        
        # Visualiser
        labels = ['Cross-Metamodel', 'Within-Metamodel']
        diffs = [cross_diff, within_diff]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, diffs, color=['salmon', 'lightgreen'])
        
        plt.title('Effet différentiel des embeddings par type de transformation')
        plt.ylabel('Différence en % par rapport à la baseline')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Ajouter les valeurs
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.1 if height >= 0 else height - 0.3,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        plt.savefig('results/embedding_differential_effect.png')
        plt.show()

    def visualize_token_pair_similarity(self, source_pairs, target_pairs):
        """
        Visualizes the similarity matrix between source and target token pairs.
        
        Args:
            source_pairs: List of source token pairs
            target_pairs: List of target token pairs
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not source_pairs or not target_pairs:
            print("Cannot visualize token pair similarity: No token pairs provided")
            return
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(source_pairs), len(target_pairs)))
        
        for i, source_pair in enumerate(source_pairs):
            for j, target_pair in enumerate(target_pairs):
                similarity = self.similarity_calculator.calculate_similarity(source_pair, target_pair)
                similarity_matrix[i, j] = similarity
        
        # Create labels
        source_labels = [f"{tp.element_name} ({tp.element_type})" for tp in source_pairs]
        target_labels = [f"{tp.element_name} ({tp.element_type})" for tp in target_pairs]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(similarity_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Similarity')
        
        # Add labels
        plt.xticks(np.arange(len(target_labels)), target_labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(source_labels)), source_labels)
        
        plt.title('Token Pair Similarity Matrix')
        plt.tight_layout()
        
        # Save if output directory exists
        try:
            import os
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig('results/token_pair_similarity.png')
        except Exception as e:
            print(f"Could not save visualization: {e}")
        
        plt.show()

    def visualize_semantic_gaps(self, gaps):
        """
        Visualizes semantic gaps in the transformation.
        
        Args:
            gaps: List of semantic gaps, each as (source_pair, similarity)
        """
        import matplotlib.pyplot as plt
        
        if not gaps:
            print("No semantic gaps identified.")
            return
        
        # Sort gaps by similarity (ascending)
        gaps = sorted(gaps, key=lambda x: x[1])
        
        # Extract data for visualization
        pairs = [f"{g[0].element_name} ({g[0].element_type})" for g in gaps]
        similarities = [g[1] for g in gaps]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(pairs, similarities, color='salmon')
        
        # Add similarity values
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{similarities[i]:.2f}', va='center')
        
        plt.xlabel('Similarity Score')
        plt.title('Semantic Gaps in Transformation')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        plt.legend()
        plt.xlim(0, 1)
        plt.tight_layout()
        
        # Save if output directory exists
        try:
            import os
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig('results/semantic_gaps.png')
        except Exception as e:
            print(f"Could not save visualization: {e}")
        
        plt.show()