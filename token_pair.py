# token_pair.py
import re
import logging
import numpy as np

logger = logging.getLogger("TokenPair")

class TokenPair:
    """
    Représente une paire token (élément de modèle, élément de métamodèle).
    """
    def __init__(self, element_name, element_type, metamodel_info=None):
        """
        Initialise une paire token.
        
        Args:
            element_name: Nom de l'élément
            element_type: Type de l'élément
            metamodel_info: Informations sur le métamodèle
        """
        self.element_name = element_name
        self.element_type = element_type
        self.metamodel_info = metamodel_info
        self.meta_element_category = self._determine_category(element_type, metamodel_info)
    
    def _determine_category(self, element_type, metamodel_info):
        """
        Détermine la catégorie d'élément méta à partir du type d'élément.
        
        Args:
            element_type: Type de l'élément
            metamodel_info: Informations sur le métamodèle
            
        Returns:
            Catégorie de l'élément méta
        """
        if metamodel_info and 'types' in metamodel_info:
            return metamodel_info['types'].get(element_type, 'GenericElement')
        return 'GenericElement'
    
    def __str__(self):
        """Représentation en chaîne de caractères."""
        return f"{self.element_name}: {self.element_type} ({self.meta_element_category})"

class TokenPairExtractor:
    """
    Extracteur de paires token à partir de texte de modèle.
    """
    def __init__(self):
        """
        Initialise l'extracteur.
        """
        pass
    
    def extract_from_text(self, text, metamodel_info):
        """
        Extrait les paires token à partir du texte du modèle.
        
        Args:
            text: Texte du modèle
            metamodel_info: Informations sur le métamodèle
            
        Returns:
            Liste de paires token
        """
        if not text:
            logger.warning("Texte de modèle vide")
            return []
        
        token_pairs = []
        
        # Traite les lignes du texte
        for line in text.split('\n'):
            line = line.strip()
            
            # Ignorer les lignes vides et les commentaires
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Format principal: "element: type"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    element_name = parts[0].strip()
                    element_type = parts[1].strip()
                    
                    # Ignorer les lignes avec des informations incomplètes
                    if not element_name or not element_type:
                        continue
                    
                    token_pair = TokenPair(element_name, element_type, metamodel_info)
                    token_pairs.append(token_pair)
            
            # Format alternatif: "type name="element""
            else:
                ecore_match = re.search(r'(E[A-Za-z]+)\s+name="([^"]+)"', line)
                if ecore_match:
                    element_type = ecore_match.group(1)
                    element_name = ecore_match.group(2)
                    token_pair = TokenPair(element_name, element_type, metamodel_info)
                    token_pairs.append(token_pair)
                    continue
                
                uml_match = re.search(r'(Class|Attribute|Operation)\s+name="([^"]+)"', line)
                if uml_match:
                    element_type = uml_match.group(1)
                    element_name = uml_match.group(2)
                    token_pair = TokenPair(element_name, element_type, metamodel_info)
                    token_pairs.append(token_pair)
        
        if not token_pairs:
            logger.warning(f"Aucune paire token extraite du texte: {text[:100]}...")
        else:
            logger.debug(f"Extrait {len(token_pairs)} paires token")
        
        return token_pairs

class TokenPairSimilarityCalculator:
    """
    Calculateur de similarité entre paires token.
    """
    def __init__(self, embedding_model=None):
        """
        Initialise le calculateur.
        """
        self.embedding_model = embedding_model
        
        # Mappings de compatibilité pour les transformations cross-metamodel
        self.compatibility_map = {
            # UML vers Relationnel
            'Class': ['Table', 'Entity'],
            'Attribute': ['Column', 'Field', 'Property'],
            'Operation': ['Procedure', 'Function', 'Method', 'Trigger'],
            'Parameter': ['Parameter', 'Argument'],
            'Constraint': ['Constraint', 'PrimaryKey', 'ForeignKey', 'Check'],
            
            # Ecore vers UML
            'EClass': ['Class', 'Interface'],
            'EAttribute': ['Attribute', 'Property'],
            'EReference': ['Association', 'Dependency'],
            'EOperation': ['Operation', 'Method'],
            'EParameter': ['Parameter'],
            'EPackage': ['Package', 'Model'],
            'EEnum': ['Enumeration'],
            'EEnumLiteral': ['EnumerationLiteral'],
            'EDataType': ['DataType', 'PrimitiveType'],
            
            # UML vers Ecore
            'Class': ['EClass'],
            'Interface': ['EClass'],
            'Attribute': ['EAttribute'],
            'Property': ['EAttribute', 'EReference'],
            'Association': ['EReference'],
            'Operation': ['EOperation'],
            'Parameter': ['EParameter'],
            'Package': ['EPackage'],
            'Enumeration': ['EEnum'],
            'EnumerationLiteral': ['EEnumLiteral'],
            'DataType': ['EDataType'],
            
            # Relationnel
            'Table': ['Class', 'Entity'],
            'Column': ['Attribute', 'Property', 'Field'],
            'PrimaryKey': ['Constraint', 'ID'],
            'ForeignKey': ['Association', 'Reference']
        }
    
    def calculate_similarity(self, source_pair, target_pair):
        """
        Calcule la similarité entre deux paires token.
        
        Args:
            source_pair: Paire token source
            target_pair: Paire token cible
            
        Returns:
            Score de similarité entre 0 et 1
        """
        # Similarité lexicale (pondération: 0.3)
        name_sim = self._name_similarity(source_pair.element_name, target_pair.element_name)
        
        # Similarité de type (pondération: 0.5)
        type_sim = self._type_similarity(source_pair, target_pair)
        
        # Similarité contextuelle (embedding) (pondération: 0.2)
        context_sim = self._context_similarity(source_pair, target_pair)
        
        # Somme pondérée
        return 0.3 * name_sim + 0.5 * type_sim + 0.2 * context_sim
    
    def _name_similarity(self, name1, name2):
        """
        Calcule la similarité lexicale entre deux noms d'éléments.
        
        Args:
            name1: Nom du premier élément
            name2: Nom du deuxième élément
            
        Returns:
            Score de similarité entre 0 et 1
        """
        # Normalisation des noms
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        
        # Si les noms sont identiques après normalisation
        if n1 == n2:
            return 1.0
        
        # Si un nom est contenu dans l'autre
        if n1 in n2 or n2 in n1:
            return 0.8
        
        # Calcule la distance de Levenshtein normalisée
        distance = self._levenshtein_distance(n1, n2)
        max_len = max(len(n1), len(n2))
        
        if max_len == 0:
            return 0.0
        
        return 1.0 - (distance / max_len)
    
    def _normalize_name(self, name):
        """
        Normalise un nom d'élément pour la comparaison.
        
        Args:
            name: Nom à normaliser
            
        Returns:
            Nom normalisé
        """
        if not name:
            return ""
        
        # Convertit en minuscules
        name = name.lower()
        
        # Supprime les préfixes/suffixes courants
        prefixes = ['get', 'set', 'is', 'has', 'e', 'i']
        for prefix in prefixes:
            if name.startswith(prefix) and len(name) > len(prefix) + 1:
                # Vérifie si la lettre après le préfixe est majuscule (dans le nom original)
                if name[len(prefix)].isalpha():
                    name = name[len(prefix):]
        
        # Convertit les séparateurs en espaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Supprime les espaces multiples
        name = ' '.join(name.split())
        
        return name
    
    def _levenshtein_distance(self, s1, s2):
        """
        Calcule la distance de Levenshtein entre deux chaînes.
        
        Args:
            s1: Première chaîne
            s2: Deuxième chaîne
            
        Returns:
            Distance de Levenshtein
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _type_similarity(self, source_pair, target_pair):
        """
        Calcule la similarité de type entre deux paires token.
        
        Args:
            source_pair: Paire token source
            target_pair: Paire token cible
            
        Returns:
            Score de similarité entre 0 et 1
        """
        # Si les types sont identiques
        if source_pair.element_type == target_pair.element_type:
            return 1.0
        
        # Si les catégories méta sont identiques
        if source_pair.meta_element_category == target_pair.meta_element_category:
            return 0.8
        
        # Vérifie la compatibilité via la table de mappings
        if source_pair.element_type in self.compatibility_map and target_pair.element_type in self.compatibility_map.get(source_pair.element_type, []):
            return 0.6
        
        # Compatibilité inverse
        if target_pair.element_type in self.compatibility_map and source_pair.element_type in self.compatibility_map.get(target_pair.element_type, []):
            return 0.6
        
        # Similarité lexicale des types
        return 0.2 * self._name_similarity(source_pair.element_type, target_pair.element_type)
    
    def _context_similarity(self, source_pair, target_pair):
        """
        Calcule la similarité contextuelle entre deux paires token.
        
        Args:
            source_pair: Paire token source
            target_pair: Paire token cible
            
        Returns:
            Score de similarité entre 0 et 1
        """
        if self.embedding_model is None:
            # Sans modèle d'embedding, utiliser une heuristique simple
            if source_pair.element_type == target_pair.element_type:
                return 1.0
            elif source_pair.meta_element_category == target_pair.meta_element_category:
                return 0.7
            return 0.3
        
        try:
            # Construire des descriptions textuelles
            source_desc = f"{source_pair.element_name} {source_pair.element_type}"
            target_desc = f"{target_pair.element_name} {target_pair.element_type}"
            
            # Obtenir les embeddings
            source_emb = self.embedding_model.embed_text(source_desc)
            target_emb = self.embedding_model.embed_text(target_desc)
            
            # Calculer la similarité cosinus
            dot_product = np.dot(source_emb, target_emb)
            norm_source = np.linalg.norm(source_emb)
            norm_target = np.linalg.norm(target_emb)
            
            if norm_source == 0 or norm_target == 0:
                return 0.5
            
            similarity = dot_product / (norm_source * norm_target)
            
            # Normaliser à [0, 1]
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la similarité contextuelle: {str(e)}")
            return 0.5