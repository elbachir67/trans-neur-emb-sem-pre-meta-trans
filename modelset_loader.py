# modelset_loader.py
import os
import glob
import re
import logging
from collections import defaultdict

logger = logging.getLogger("ModelSetLoader")

class ModelSetLoader:
    """
    Enhanced class for loading Ecore and UML models from the ModelSet dataset.
    """
    def __init__(self, txt_folder="../txt", max_depth=5):
        """
        Initialize the loader.
        
        Args:
            txt_folder: Path to the folder containing model files
            max_depth: Maximum search depth in subdirectories
        """
        self.txt_folder = txt_folder
        self.max_depth = max_depth
        self.model_extensions = ['.txt', '.ecore', '.uml', '.xmi']
        
    def find_model_files(self):
        """
        Search for all Ecore and UML model files in the folder.
        
        Returns:
            Dictionary of file paths by metamodel type
        """
        logger.info(f"Searching for model files in {self.txt_folder}...")
        
        model_files = {
            'Ecore': [],
            'UML': [],
            'Unknown': []
        }
        
        # Build patterns for file type recognition
        ecore_patterns = [r'\.ecore$', r'ecore', r'emf']
        uml_patterns = [r'\.uml$', r'uml', r'aurora']
        
        # Recursive directory traversal
        for root, dirs, files in os.walk(self.txt_folder):
            # Limit depth
            depth = root[len(self.txt_folder):].count(os.sep)
            if depth > self.max_depth:
                dirs.clear()  # Don't go deeper
                continue
            
            # Filter files with relevant extensions
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                path_lower = file_path.lower()
                
                # Check if the file has a relevant extension
                if any(file_lower.endswith(ext) for ext in self.model_extensions):
                    try:
                        # Determine model type
                        model_type = 'Unknown'
                        
                        # Check if it's an Ecore model
                        if any(re.search(pattern, path_lower) for pattern in ecore_patterns):
                            model_type = 'Ecore'
                        # Check if it's a UML model
                        elif any(re.search(pattern, path_lower) for pattern in uml_patterns):
                            model_type = 'UML'
                        # If type is not clear from the name, check content
                        else:
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    content = f.read(1000)  # Read the first 1000 characters
                                    if 'ecore' in content.lower() or 'eclass' in content.lower():
                                        model_type = 'Ecore'
                                    elif 'uml' in content.lower() or '<class' in content.lower():
                                        model_type = 'UML'
                            except Exception as e:
                                logger.warning(f"Cannot read file {file_path}: {str(e)}")
                        
                        # Add the file to the corresponding list
                        model_files[model_type].append(file_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Display statistics
        logger.info(f"Found {len(model_files['Ecore'])} Ecore models, {len(model_files['UML'])} UML models, and {len(model_files['Unknown'])} unknown models")
        
        return model_files
    
    def load_models(self, max_models=None):
        """
        Load Ecore and UML models.
        
        Args:
            max_models: Maximum number of models to load per type
            
        Returns:
            Dictionary of loaded models
        """
        model_files = self.find_model_files()
        loaded_models = {
            'Ecore': {},
            'UML': {}
        }
        
        # Load Ecore models
        for i, file_path in enumerate(model_files['Ecore']):
            if max_models is not None and i >= max_models:
                break
                
            try:
                model_id = f"Ecore_{i}"
                model_content = self._read_model_file(file_path)
                
                if model_content:
                    # Preprocess content
                    preprocessed_content = self.preprocess_model_text(model_content)
                    
                    # Determine domain
                    domain = self.determine_domain(file_path, model_content)
                    
                    loaded_models['Ecore'][model_id] = {
                        'id': model_id,
                        'file_path': file_path,
                        'content': model_content,
                        'preprocessed': preprocessed_content,
                        'metamodel': 'Ecore',
                        'domain': domain
                    }
                    
                    logger.debug(f"Loaded Ecore model {model_id} from {file_path}")
            except Exception as e:
                logger.error(f"Error loading Ecore model {file_path}: {str(e)}")
        
        # Load UML models
        for i, file_path in enumerate(model_files['UML']):
            if max_models is not None and i >= max_models:
                break
                
            try:
                model_id = f"UML_{i}"
                model_content = self._read_model_file(file_path)
                
                if model_content:
                    # Preprocess content
                    preprocessed_content = self.preprocess_model_text(model_content)
                    
                    # Determine domain
                    domain = self.determine_domain(file_path, model_content)
                    
                    loaded_models['UML'][model_id] = {
                        'id': model_id,
                        'file_path': file_path,
                        'content': model_content,
                        'preprocessed': preprocessed_content,
                        'metamodel': 'UML',
                        'domain': domain
                    }
                    
                    logger.debug(f"Loaded UML model {model_id} from {file_path}")
            except Exception as e:
                logger.error(f"Error loading UML model {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(loaded_models['Ecore'])} Ecore models and {len(loaded_models['UML'])} UML models")
        
        return loaded_models
    
    def _read_model_file(self, file_path):
        """
        Read a model file with error handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content, or None in case of error
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
                # Check if the content is valid
                if len(content.strip()) < 10:
                    logger.warning(f"File ignored because it's too short: {file_path}")
                    return None
                
                return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def preprocess_model_text(self, text):
        """
        Preprocess model text for semantic analysis with enhanced support
        for different formats.
        
        Args:
            text: Raw model text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Detect file format
        format_type = self._detect_format(text)
        
        # Process according to format
        if format_type == 'xml':
            return self._process_xml_format(text)
        elif format_type == 'text':
            text_result = self._process_text_format(text)
            # If standard text processing didn't yield anything, try simple format
            if not text_result:
                return self._process_simple_format(text)
            return text_result
        else:
            # Unknown format, try multiple approaches
            xml_result = self._process_xml_format(text)
            text_result = self._process_text_format(text)
            simple_result = self._process_simple_format(text)
            
            # Choose version that extracted most information
            xml_lines = len(xml_result.split('\n')) if xml_result else 0
            text_lines = len(text_result.split('\n')) if text_result else 0
            simple_lines = len(simple_result.split('\n')) if simple_result else 0
            
            if xml_lines >= text_lines and xml_lines >= simple_lines:
                return xml_result
            elif text_lines >= simple_lines:
                return text_result
            else:
                return simple_result
    
    def _detect_format(self, text):
        """
        Detect the format of the text (XML/XMI or plain text).
        
        Args:
            text: Text to analyze
            
        Returns:
            'xml', 'text' or 'unknown'
        """
        # Check for XML tags
        if '<' in text and '>' in text and ('<?xml' in text or '<ecore:' in text or '<uml:' in text or '<xmi:' in text):
            return 'xml'
        
        # Check if the text looks like a word list file
        words = text.split()
        if len(words) > 5 and 'model' in words:
            # This seems to be the specific format we've seen
            return 'text'
        
        # Standard format with lines of the form "name: type"
        if ':' in text and '<' not in text:
            return 'text'
        
        return 'unknown'
    
    def _process_xml_format(self, text):
        """
        Process text in XML/XMI format with improved element detection.
        
        Args:
            text: XML/XMI text
            
        Returns:
            Preprocessed text
        """
        lines = []
        
        # Display the first characters for debugging
        logger.debug(f"Start of XML content: {text[:100]}")
        
        # Patterns for elements with attributes
        patterns = [
            # Standard pattern for elements with name attribute
            r'<([^/\s>]+)[^>]*\s+name="([^"]+)"',
            
            # Pattern for UML elements in GenMyModel files
            r'<packagedElement\s+xmi:type="uml:([^"]+)"\s+[^>]*\s+name="([^"]+)"',
            
            # Pattern for elements with xmi:id
            r'<([^/\s>]+)[^>]*\s+xmi:id="([^"]+)"',
            
            # Pattern for class attributes
            r'<ownedAttribute\s+[^>]*\s+name="([^"]+)"[^>]*\s+type="([^"]+)"',
            
            # Pattern for operations
            r'<ownedOperation\s+[^>]*\s+name="([^"]+)"',
            
            # Pattern for parameters
            r'<ownedParameter\s+[^>]*\s+name="([^"]+)"'
        ]
        
        # Search for all matches for each pattern
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    if pattern == r'<packagedElement\s+xmi:type="uml:([^"]+)"\s+[^>]*\s+name="([^"]+)"':
                        # For UML elements in GenMyModel
                        element_type = match[0]
                        element_name = match[1]
                    elif pattern == r'<ownedAttribute\s+[^>]*\s+name="([^"]+)"[^>]*\s+type="([^"]+)"':
                        # For attributes with their type
                        element_name = match[0]
                        element_type = "Attribute"
                    elif pattern == r'<ownedOperation\s+[^>]*\s+name="([^"]+)"':
                        # For operations
                        element_name = match[0]
                        element_type = "Operation"
                    elif pattern == r'<ownedParameter\s+[^>]*\s+name="([^"]+)"':
                        # For parameters
                        element_name = match[0]
                        element_type = "Parameter"
                    else:
                        # For generic patterns
                        element_type = match[0].split(':')[-1]
                        element_name = match[1]
                    
                    # Filter out irrelevant elements
                    if element_type.lower() not in ['xml', 'xmi', 'documentation', 'annotation']:
                        lines.append(f"{element_name}: {element_type}")
        
        # Add specific search for associations
        association_pattern = r'<packagedElement\s+xmi:type="uml:Association"[^>]*>'
        association_matches = re.findall(association_pattern, text)
        for i, _ in enumerate(association_matches):
            # Use a generic name for unnamed associations
            lines.append(f"Association_{i}: Association")
        
        # Search for generalizations
        generalization_pattern = r'<generalization\s+[^>]*\s+general="([^"]+)"'
        generalization_matches = re.findall(generalization_pattern, text)
        for i, general in enumerate(generalization_matches):
            lines.append(f"Generalization_{i}: Generalization")
        
        if not lines:
            # If no elements were found with previous patterns,
            # try a more generic approach to extract information
            generic_pattern = r'<([^/\s>:]+)[^>]*\s+([a-zA-Z]+)="([^"]+)"'
            generic_matches = re.findall(generic_pattern, text)
            
            for match in generic_matches:
                element_type = match[0]
                attribute_name = match[1]
                attribute_value = match[2]
                
                if attribute_name == 'name':
                    lines.append(f"{attribute_value}: {element_type}")
        
        result = '\n'.join(lines)
        logger.debug(f"Preprocessed text (XML): {result[:100]}...")
        return result
    
    def _process_text_format(self, text):
        """
        Process text in plain text format.
        
        Args:
            text: Plain text
            
        Returns:
            Preprocessed text
        """
        lines = []
        
        # Patterns for different text formats
        patterns = [
            (r'^\s*([^:]+):\s*([^\s]+)\s*$', lambda m: (m.group(1).strip(), m.group(2).strip())),  # Name: Type
            (r'(E[A-Za-z]+|Class|Attribute|Operation)\s+name="([^"]+)"', lambda m: (m.group(2), m.group(1))),  # Type name="Name"
            (r'([A-Za-z0-9_]+)\s+:\s+([A-Za-z0-9_]+)', lambda m: (m.group(1).strip(), m.group(2).strip())),  # Name : Type
            (r'@([A-Za-z0-9_]+)\(([A-Za-z0-9_]+)\)', lambda m: (m.group(2), m.group(1)))  # @Type(Name)
        ]
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Try each pattern
            for pattern, extractor in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    try:
                        element_name, element_type = extractor(match)
                        if element_name and element_type:
                            lines.append(f"{element_name}: {element_type}")
                    except Exception as e:
                        logger.debug(f"Error in extraction with pattern {pattern}: {str(e)}")
        
        return '\n'.join(lines)
    
    def _process_simple_format(self, text):
        """
        Process text in simple format, without XML tags.
        
        Args:
            text: Simple text
            
        Returns:
            Preprocessed text
        """
        lines = []
        words = text.split()
        
        # Identify activity models
        if 'Activity' in words:
            lines.append("Activity: Activity")
            
            # Look for control flows (ControlFlow)
            i = 0
            while i < len(words):
                if words[i].startswith('ControlFlow'):
                    if i + 1 < len(words) and words[i+1] == 'true':
                        lines.append(f"{words[i]}: ControlFlow")
                        i += 2  # Skip next word (true)
                    elif i + 1 < len(words):
                        lines.append(f"{words[i]}: ControlFlow")
                        i += 2  # Skip next word
                    else:
                        lines.append(f"{words[i]}: ControlFlow")
                        i += 1
                else:
                    i += 1
            
            # Look for other common activity diagram elements
            activities = ['Initial', 'Final', 'Action', 'Decision', 'Merge', 'Fork', 'Join']
            
            for activity in activities:
                if activity in words:
                    index = words.index(activity)
                    if index > 0 and not words[index-1].startswith('Control'):
                        name = words[index-1]
                        lines.append(f"{name}: {activity}")
        
        # Process class elements if present
        if 'Class' in words:
            index = words.index('Class')
            if index > 0:
                class_name = words[index-1]
                lines.append(f"{class_name}: Class")
                
                # Look for potential attributes and methods
                for i, word in enumerate(words):
                    if word == 'attribute' and i + 1 < len(words):
                        lines.append(f"{words[i+1]}: Attribute")
                    elif word == 'method' and i + 1 < len(words):
                        lines.append(f"{words[i+1]}: Operation")
        
        # Try to extract name-type pairs
        i = 0
        while i < len(words) - 1:
            current_word = words[i]
            next_word = words[i+1]
            
            # If the current word starts with uppercase and the next one too, 
            # it might be an element and its type
            if (current_word and next_word and 
                len(current_word) > 1 and len(next_word) > 1 and 
                current_word[0].isupper() and next_word[0].isupper() and 
                next_word not in ['true', 'false']):
                
                lines.append(f"{current_word}: {next_word}")
                i += 2
            else:
                i += 1
        
        # If no elements were extracted, try a more generic approach
        if not lines:
            # Extract all words that start with uppercase
            capital_words = [w for w in words if w and len(w) > 1 and w[0].isupper()]
            
            # For each word, try to determine an appropriate type
            for word in capital_words:
                if word.startswith('ControlFlow'):
                    lines.append(f"{word}: ControlFlow")
                elif word in ['Activity', 'Class', 'Package', 'State', 'Sequence']:
                    lines.append(f"{word}: Diagram")
                elif word in ['true', 'false']:
                    continue  # Ignore booleans
                elif 'Action' in word or word.endswith('Action'):
                    lines.append(f"{word}: Action")
                else:
                    # By default, consider as a generic element
                    lines.append(f"{word}: Element")
        
        result = '\n'.join(lines)
        logger.debug(f"Preprocessed text (simple format): {result[:100]}...")
        return result
    
    def determine_domain(self, path, content):
        """
        Determine the domain of a model from the path and content.
        
        Args:
            path: File path
            content: File content
            
        Returns:
            Domain name
        """
        domain = 'Unknown'
        path_lower = path.lower()
        content_lower = content.lower()[:2000]  # Analyze only the beginning of the content
        
        # Dictionary of keywords to determine the domain
        domain_keywords = {
            'finance': ['finance', 'bank', 'accounting', 'money', 'payment', 'transaction'],
            'healthcare': ['health', 'medical', 'patient', 'hospital', 'doctor', 'clinic', 'care'],
            'library': ['library', 'book', 'author', 'publication', 'borrow', 'catalog'],
            'education': ['school', 'university', 'student', 'teacher', 'course', 'class', 'education'],
            'retail': ['shop', 'store', 'product', 'ecommerce', 'customer', 'order', 'retail'],
            'travel': ['travel', 'airline', 'flight', 'hotel', 'booking', 'reservation', 'trip', 'tourism'],
            'insurance': ['insurance', 'policy', 'claim', 'risk', 'coverage', 'premium'],
            'telecom': ['telecom', 'network', 'communication', 'phone', 'mobile', 'cellular'],
            'manufacturing': ['manufacturing', 'factory', 'production', 'assembly', 'industrial'],
            'automotive': ['car', 'vehicle', 'automotive', 'engine', 'driver', 'transport']
        }
        
        # Search in path and content
        for domain_name, keywords in domain_keywords.items():
            # Check path
            if any(keyword in path_lower for keyword in keywords):
                return domain_name.capitalize()
            
            # Check content
            keyword_count = sum(content_lower.count(keyword) for keyword in keywords)
            if keyword_count > 3:  # Arbitrary threshold to avoid false positives
                return domain_name.capitalize()
        
        # If no domain is detected, try to extract additional information
        if 'class' in content_lower and 'attribute' in content_lower:
            if 'order' in content_lower and 'customer' in content_lower:
                return 'Retail'
            elif 'student' in content_lower and 'course' in content_lower:
                return 'Education'
            elif 'patient' in content_lower and 'doctor' in content_lower:
                return 'Healthcare'
            elif 'book' in content_lower and 'author' in content_lower:
                return 'Library'
        
        return domain
    
    def infer_metamodel_info(self, metamodel_name):
        """
        Deduce metamodel information from the metamodel name.
        
        Args:
            metamodel_name: Metamodel name
            
        Returns:
            Dictionary with metamodel information
        """
        if metamodel_name == 'Ecore':
            return {
                'name': 'Ecore',
                'types': {
                    'EClass': 'ClassElement',
                    'EAttribute': 'PropertyElement',
                    'EReference': 'RelationElement',
                    'EOperation': 'BehaviorElement',
                    'EParameter': 'ParameterElement',
                    'EPackage': 'PackageElement',
                    'EAnnotation': 'AnnotationElement',
                    'EEnum': 'EnumerationElement',
                    'EEnumLiteral': 'LiteralElement',
                    'EDataType': 'DataElement',
                    'EClassifier': 'ClassElement',
                    # Added additional types
                    'EObject': 'ClassElement',
                    'EModelElement': 'ClassElement',
                    'EFactory': 'ClassElement',
                    'EStructuralFeature': 'PropertyElement',
                    'ETypedElement': 'PropertyElement'
                }
            }
        elif metamodel_name == 'UML':
            return {
                'name': 'UML',
                'types': {
                    'Class': 'ClassElement',
                    'Attribute': 'PropertyElement',
                    'Property': 'PropertyElement',
                    'Operation': 'BehaviorElement',
                    'Parameter': 'ParameterElement',
                    'Constraint': 'ConstraintElement',
                    'Package': 'PackageElement',
                    'Association': 'RelationElement',
                    'Dependency': 'RelationElement',
                    'Generalization': 'RelationElement',
                    'Interface': 'ClassElement',
                    'Enumeration': 'EnumerationElement',
                    'EnumerationLiteral': 'LiteralElement',
                    'DataType': 'DataElement',
                    'Stereotype': 'ClassElement',
                    'Model': 'PackageElement',
                    # Added additional types
                    'Actor': 'ClassElement',
                    'UseCase': 'BehaviorElement',
                    'Activity': 'BehaviorElement',
                    'State': 'BehaviorElement',
                    'Transition': 'RelationElement',
                    'Component': 'ClassElement',
                    'Node': 'ClassElement',
                    'Artifact': 'ClassElement',
                    'Comment': 'AnnotationElement',
                    # Activity types
                    'ControlFlow': 'RelationElement',
                    'InitialNode': 'BehaviorElement',
                    'FinalNode': 'BehaviorElement',
                    'MergeNode': 'BehaviorElement',
                    'DecisionNode': 'BehaviorElement',
                    'Action': 'BehaviorElement',
                    'ActivityNode': 'BehaviorElement',
                    'ActivityEdge': 'RelationElement',
                    'ForkNode': 'BehaviorElement',
                    'JoinNode': 'BehaviorElement'
                }
            }
        elif metamodel_name == 'RDBMS':
            return {
                'name': 'RDBMS',
                'types': {
                    'Table': 'RelationalElement',
                    'Column': 'RelationalElement',
                    'PrimaryKey': 'ConstraintElement',
                    'ForeignKey': 'ConstraintElement',
                    'Schema': 'PackageElement',
                    'Index': 'RelationalElement',
                    'Trigger': 'BehaviorElement',
                    'StoredProcedure': 'BehaviorElement',
                    'Constraint': 'ConstraintElement',
                    'View': 'RelationalElement',
                    'Database': 'PackageElement',
                    'Check': 'ConstraintElement',
                    'Unique': 'ConstraintElement',
                    'NotNull': 'ConstraintElement'
                }
            }
        else:
            return {
                'name': 'Generic',
                'types': {
                    'Entity': 'ClassElement',
                    'Property': 'PropertyElement',
                    'Relation': 'RelationElement',
                    'Operation': 'BehaviorElement',
                    'Package': 'PackageElement',
                    'Parameter': 'ParameterElement',
                    'Constraint': 'ConstraintElement',
                    'Annotation': 'AnnotationElement',
                    'Enumeration': 'EnumerationElement',
                    'Literal': 'LiteralElement',
                    'DataType': 'DataElement',
                    'Element': 'GenericElement'
                }
            }
    
    def create_model_pairs(self, max_pairs=20):
        """
        Create model transformation pairs.
        
        Args:
            max_pairs: Maximum number of pairs to create
            
        Returns:
            List of model pairs
        """
        logger.info(f"Creating model pairs (max: {max_pairs})...")
        
        # Load models
        models = self.load_models(max_models=50)  # Load more models for a good sample
        
        if not models or (not models['Ecore'] and not models['UML']):
            logger.error("No models loaded. Cannot create pairs.")
            return []
        
        # Group models by domain
        domain_groups = defaultdict(lambda: {'Ecore': [], 'UML': []})
        
        for metamodel, model_dict in models.items():
            for model_id, model_info in model_dict.items():
                domain = model_info['domain']
                domain_groups[domain][metamodel].append(model_id)
        
        # Creation of pairs
        model_pairs = []
        
        # 1. Cross-metamodel pairs (Ecore to UML)
        cross_pairs_count = 0
        max_cross_pairs = int(max_pairs * 0.9)  # 90% cross-metamodel
        
        # First, try to make pairs in the same domain
        for domain, domain_models in domain_groups.items():
            ecore_models = domain_models['Ecore']
            uml_models = domain_models['UML']
            
            # Create pairs in the same domain
            for i in range(min(len(ecore_models), min(len(uml_models), max_cross_pairs - cross_pairs_count))):
                if cross_pairs_count >= max_cross_pairs:
                    break
                    
                source_id = ecore_models[i]
                target_id = uml_models[i]
                
                source_model = models['Ecore'][source_id]
                target_model = models['UML'][target_id]
                
                # Get metamodels
                source_metamodel = self.infer_metamodel_info('Ecore')
                target_metamodel = self.infer_metamodel_info('UML')
                
                # Deduce transformation rules
                rules = self.infer_rules_from_metamodels(source_metamodel, target_metamodel)
                
                model_pairs.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'source_text': source_model['preprocessed'],
                    'target_text': target_model['preprocessed'],
                    'source_domain': domain,
                    'target_domain': domain,
                    'is_cross_metamodel': True,
                    'source_metamodel': source_metamodel,
                    'target_metamodel': target_metamodel,
                    'rules': rules
                })
                
                cross_pairs_count += 1
        
        # If we don't have enough pairs in the same domain, create cross-domain pairs
        if cross_pairs_count < max_cross_pairs:
            all_ecore_ids = list(models['Ecore'].keys())
            all_uml_ids = list(models['UML'].keys())
            
            # Shuffle lists for better diversity
            import random
            random.shuffle(all_ecore_ids)
            random.shuffle(all_uml_ids)
            
            for i in range(min(len(all_ecore_ids), min(len(all_uml_ids), max_cross_pairs - cross_pairs_count))):
                source_id = all_ecore_ids[i]
                target_id = all_uml_ids[i]
                
                # Avoid duplications
                if any(p['source_id'] == source_id and p['target_id'] == target_id for p in model_pairs):
                    continue
                
                source_model = models['Ecore'][source_id]
                target_model = models['UML'][target_id]
                
                # Get metamodels
                source_metamodel = self.infer_metamodel_info('Ecore')
                target_metamodel = self.infer_metamodel_info('UML')
                
                # Deduce transformation rules
                rules = self.infer_rules_from_metamodels(source_metamodel, target_metamodel)
                
                model_pairs.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'source_text': source_model['preprocessed'],
                    'target_text': target_model['preprocessed'],
                    'source_domain': source_model['domain'],
                    'target_domain': target_model['domain'],
                    'is_cross_metamodel': True,
                    'source_metamodel': source_metamodel,
                    'target_metamodel': target_metamodel,
                    'rules': rules
                })
                
                cross_pairs_count += 1
        
        # 2. Within-metamodel pairs (UML to UML and Ecore to Ecore)
        within_pairs_count = 0
        max_within_pairs = max_pairs - cross_pairs_count
        
        # For each domain, create within-metamodel pairs
        for domain, domain_models in domain_groups.items():
            if within_pairs_count >= max_within_pairs:
                break
                
            # Ecore to Ecore pairs
            ecore_models = domain_models['Ecore']
            for i in range(min(len(ecore_models) - 1, max_within_pairs - within_pairs_count)):
                source_id = ecore_models[i]
                target_id = ecore_models[i + 1]
                
                source_model = models['Ecore'][source_id]
                target_model = models['Ecore'][target_id]
                
                source_metamodel = self.infer_metamodel_info('Ecore')
                target_metamodel = source_metamodel  # Same metamodel
                
                rules = self.infer_rules_from_metamodels(source_metamodel, target_metamodel)
                
                model_pairs.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'source_text': source_model['preprocessed'],
                    'target_text': target_model['preprocessed'],
                    'source_domain': domain,
                    'target_domain': domain,
                    'is_cross_metamodel': False,
                    'source_metamodel': source_metamodel,
                    'target_metamodel': target_metamodel,
                    'rules': rules
                })
                
                within_pairs_count += 1
                
                if within_pairs_count >= max_within_pairs:
                    break
            
            if within_pairs_count >= max_within_pairs:
                break
                
            # UML to UML pairs
            uml_models = domain_models['UML']
            for i in range(min(len(uml_models) - 1, max_within_pairs - within_pairs_count)):
                source_id = uml_models[i]
                target_id = uml_models[i + 1]
                
                source_model = models['UML'][source_id]
                target_model = models['UML'][target_id]
                
                source_metamodel = self.infer_metamodel_info('UML')
                target_metamodel = source_metamodel  # Same metamodel
                
                rules = self.infer_rules_from_metamodels(source_metamodel, target_metamodel)
                
                model_pairs.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'source_text': source_model['preprocessed'],
                    'target_text': target_model['preprocessed'],
                    'source_domain': domain,
                    'target_domain': domain,
                    'is_cross_metamodel': False,
                    'source_metamodel': source_metamodel,
                    'target_metamodel': target_metamodel,
                    'rules': rules
                })
                
                within_pairs_count += 1
                
                if within_pairs_count >= max_within_pairs:
                    break
        
        logger.info(f"Created {len(model_pairs)} model pairs: {cross_pairs_count} cross-metamodel, {within_pairs_count} within-metamodel")
        
        return model_pairs
    
    def infer_rules_from_metamodels(self, source_metamodel, target_metamodel):
        """
        Deduce transformation rules from metamodels.
        
        Args:
            source_metamodel: Source metamodel information
            target_metamodel: Target metamodel information
            
        Returns:
            List of deduced transformation rules
        """
        rules = []
        
        # Element type correspondence
        mapping = {
            'ClassElement': ['ClassElement'],
            'PropertyElement': ['PropertyElement'],
            'BehaviorElement': ['BehaviorElement'],
            'RelationElement': ['RelationElement'],
            'ConstraintElement': ['ConstraintElement'],
            'PackageElement': ['PackageElement'],
            'ParameterElement': ['ParameterElement'],
            'EnumerationElement': ['EnumerationElement'],
            'LiteralElement': ['LiteralElement'],
            'DataElement': ['DataElement'],
            'AnnotationElement': ['AnnotationElement']
        }
        
        # Specific Ecore to UML mappings
        if source_metamodel['name'] == 'Ecore' and target_metamodel['name'] == 'UML':
            specific_rules = [
                'EClass must be transformed to Class',
                'EAttribute must be transformed to Property',
                'EReference must be transformed to Association',
                'EOperation must be transformed to Operation',
                'EParameter must be transformed to Parameter',
                'EPackage must be transformed to Package',
                'EEnum must be transformed to Enumeration',
                'EEnumLiteral must be transformed to EnumerationLiteral',
                'EDataType must be transformed to DataType'
            ]
            rules.extend(specific_rules)
        
        # Specific UML to Ecore mappings
        elif source_metamodel['name'] == 'UML' and target_metamodel['name'] == 'Ecore':
            specific_rules = [
                'Class must be transformed to EClass',
                'Property must be transformed to EAttribute or EReference',
                'Association must be transformed to EReference',
                'Operation must be transformed to EOperation',
                'Parameter must be transformed to EParameter',
                'Package must be transformed to EPackage',
                'Enumeration must be transformed to EEnum',
                'EnumerationLiteral must be transformed to EEnumLiteral',
                'DataType must be transformed to EDataType'
            ]
            rules.extend(specific_rules)
            
        # Specific UML to RDBMS mappings
        elif source_metamodel['name'] == 'UML' and target_metamodel['name'] == 'RDBMS':
            specific_rules = [
                'Class must be transformed to Table',
                'Attribute must be transformed to Column',
                'Operation must be transformed to StoredProcedure or Trigger',
                'Association must be transformed to ForeignKey',
                'Constraint must be preserved',
                'Primary keys must be preserved',
                'Element names must be preserved'
            ]
            rules.extend(specific_rules)
            
        # Specific Ecore to RDBMS mappings
        elif source_metamodel['name'] == 'Ecore' and target_metamodel['name'] == 'RDBMS':
            specific_rules = [
                'EClass must be transformed to Table',
                'EAttribute must be transformed to Column',
                'EReference must be transformed to ForeignKey',
                'EOperation must be transformed to StoredProcedure',
                'EPackage must be transformed to Schema',
                'Element names must be preserved'
            ]
            rules.extend(specific_rules)
        
        # Generic rules based on element categories
        else:
            for source_type, source_meta in source_metamodel['types'].items():
                if source_meta in mapping:
                    target_metas = mapping[source_meta]
                    
                    # Find corresponding target types
                    target_types = []
                    for target_type, target_meta in target_metamodel['types'].items():
                        if target_meta in target_metas:
                            target_types.append(target_type)
                    
                    if target_types:
                        target_types_str = ', '.join(target_types)
                        rules.append(f"{source_type} must be transformed to one of: {target_types_str}")
        
        # Property preservation rules
        preservation_rules = [
            "Name attributes must be preserved",
            "Type references must be updated to corresponding types",
            "Containment relationships must be preserved",
            "Multiplicity constraints must be preserved"
        ]
        rules.extend(preservation_rules)
        
        return rules
    
    def debug_file_content(self, file_path):
        """
        Displays raw file content for debugging.
        
        Args:
            file_path: Path to the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            logger.info(f"=== Content of file {file_path} ===")
            logger.info(content[:1000])  # Display first 1000 characters
            logger.info("...")
            
            # Preprocessing
            logger.info("=== After preprocessing ===")
            preprocessed = self.preprocess_model_text(content)
            logger.info(preprocessed)
            
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def visualize_model_statistics(self, max_models=20):
        """
        Visualize statistics about loaded models.
        
        Args:
            max_models: Maximum number of models to include
            
        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Load models
        models = self.load_models(max_models=max_models)
        
        # Count models by metamodel
        metamodels = {
            'Ecore': len(models['Ecore']),
            'UML': len(models['UML'])
        }
        
        # Count models by domain
        domains = defaultdict(int)
        
        for metamodel, model_dict in models.items():
            for model_id, model_info in model_dict.items():
                domains[model_info['domain']] += 1
        
        # Count elements by metamodel
        element_counts = {
            'Ecore': {},
            'UML': {}
        }
        
        for metamodel, model_dict in models.items():
            for model_id, model_info in model_dict.items():
                preprocessed = model_info['preprocessed']
                lines = preprocessed.split('\n')
                
                for line in lines:
                    if ':' in line:
                        element_type = line.split(':')[1].strip()
                        element_counts[metamodel][element_type] = element_counts[metamodel].get(element_type, 0) + 1
        
        # Figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Graph 1: Metamodel distribution
        plt.subplot(2, 2, 1)
        plt.bar(metamodels.keys(), metamodels.values(), color=['skyblue', 'salmon'])
        plt.title('Model distribution by metamodel')
        plt.ylabel('Number of models')
        
        # Graph 2: Domain distribution
        plt.subplot(2, 2, 2)
        domains_sorted = dict(sorted(domains.items(), key=lambda item: item[1], reverse=True))
        plt.bar(domains_sorted.keys(), domains_sorted.values(), color='lightgreen')
        plt.title('Model distribution by domain')
        plt.ylabel('Number of models')
        plt.xticks(rotation=45, ha='right')
        
        # Graph 3: Top 5 Ecore element types
        plt.subplot(2, 2, 3)
        if element_counts['Ecore']:
            ecore_sorted = dict(sorted(element_counts['Ecore'].items(), key=lambda item: item[1], reverse=True)[:5])
            plt.bar(ecore_sorted.keys(), ecore_sorted.values(), color='skyblue')
            plt.title('Top 5 Ecore element types')
            plt.ylabel('Number of occurrences')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, 'No Ecore elements found', ha='center', va='center')
            plt.title('Ecore element types')
        
        # Graph 4: Top 5 UML element types
        plt.subplot(2, 2, 4)
        if element_counts['UML']:
            uml_sorted = dict(sorted(element_counts['UML'].items(), key=lambda item: item[1], reverse=True)[:5])
            plt.bar(uml_sorted.keys(), uml_sorted.values(), color='salmon')
            plt.title('Top 5 UML element types')
            plt.ylabel('Number of occurrences')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, 'No UML elements found', ha='center', va='center')
            plt.title('UML element types')
        
        plt.tight_layout()
        plt.savefig('model_statistics.png')
        plt.show()
        
        # Additional information
        logger.info(f"Model statistics:")
        logger.info(f"  Total models: {sum(metamodels.values())}")
        for metamodel, count in metamodels.items():
            logger.info(f"  {metamodel}: {count} models")
        
        logger.info(f"Distribution by domain:")
        for domain, count in domains_sorted.items():
            logger.info(f"  {domain}: {count} models")