# test_modelset.py
import os
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from datetime import datetime

# Import custom modules
from semantic_preservation import SemanticPreservationFramework
from modelset_loader import ModelSetLoader

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelSetExperiment")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the semantic preservation framework with ModelSet dataset")
    
    parser.add_argument('--model-folder', type=str, default='../modelset',
                       help='Path to the folder containing model files (default: ../modelset)')
    
    parser.add_argument('--embedding-size', type=int, default=128,
                       help='Dimension of neural embeddings (default: 128)')
    
    parser.add_argument('--window-size', type=int, default=3,
                       help='Size of historical context window (default: 3)')
    
    parser.add_argument('--max-pairs', type=int, default=20,
                       help='Maximum number of model pairs to create (default: 20)')
    
    parser.add_argument('--max-models', type=int, default=50,
                       help='Maximum number of models to load per metamodel (default: 50)')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs for auto-regression model (default: 30)')
    
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Disable neural embeddings')
    
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only visualize model statistics without performing evaluation')
    
    parser.add_argument('--diagnostic', action='store_true',
                       help='Run diagnostic test on token pair extraction')
    
    parser.add_argument('--specific-file', type=str, default=None,
                    help='Path to a specific file to analyze for diagnostics')
    
    parser.add_argument('--rdbms', action='store_true',
                       help='Run specific tests for RDBMS transformations')
    
    return parser.parse_args()

def diagnostic_test(framework, loader, args=None):
    """
    Performs diagnostic test to verify token pair extraction.
    """
    logger.info("Running diagnostic test on token pair extraction...")
    
    # If a specific file is provided, analyze that file
    if args and hasattr(args, 'specific_file') and args.specific_file and os.path.exists(args.specific_file):
        logger.info(f"Analyzing specific file: {args.specific_file}")
        content = loader.debug_file_content(args.specific_file)
        
        if content:
            # Determine metamodel
            metamodel_name = 'Unknown'
            if 'ecore' in args.specific_file.lower() or 'EClass' in content:
                metamodel_name = 'Ecore'
            elif 'uml' in args.specific_file.lower() or '<uml:' in content:
                metamodel_name = 'UML'
            
            metamodel_info = loader.infer_metamodel_info(metamodel_name)
            
            # Preprocess content
            preprocessed = loader.preprocess_model_text(content)
            
            # Extract token pairs
            token_pairs = framework.extract_token_pairs(preprocessed, metamodel_info)
            
            logger.info(f"Number of token pairs extracted: {len(token_pairs)}")
            for i, tp in enumerate(token_pairs):
                logger.info(f"  {i+1}. {tp}")
        
        return
    
    # Load models
    models = loader.load_models(max_models=5)
    
    if not models or (not models['Ecore'] and not models['UML']):
        logger.error("No models loaded. Cannot run diagnostics.")
        return
    
    # Test on a few Ecore models
    logger.info("=== DIAGNOSTIC ECORE MODELS ===")
    for model_id, model in list(models['Ecore'].items())[:3]:
        try:
            logger.info(f"Model: {model_id} - {model['file_path']}")
            logger.info(f"Raw content size: {len(model['content'])} characters")
            logger.info(f"Preprocessed content size: {len(model['preprocessed'])} characters")
            logger.info(f"Preprocessed content preview: {model['preprocessed'][:200]}...")
            
            # Token pair extraction
            metamodel_info = loader.infer_metamodel_info('Ecore')
            token_pairs = framework.extract_token_pairs(model['preprocessed'], metamodel_info)
            
            logger.info(f"Number of token pairs extracted: {len(token_pairs)}")
            
            # Display first 5 token pairs
            for i, tp in enumerate(token_pairs[:5]):
                logger.info(f"  {i+1}. {tp}")
            
            # If no token pairs were extracted, try direct extraction
            if len(token_pairs) == 0:
                logger.info("Attempting direct extraction from raw content:")
                direct_preprocessed = loader._process_xml_format(model['content'])
                logger.info(f"Direct preprocessing size: {len(direct_preprocessed)} characters")
                direct_pairs = framework.extract_token_pairs(direct_preprocessed, metamodel_info)
                logger.info(f"Number of token pairs extracted directly: {len(direct_pairs)}")
                for i, tp in enumerate(direct_pairs[:5]):
                    logger.info(f"  {i+1}. {tp}")
            
            logger.info("---")
        except Exception as e:
            logger.error(f"Error during model {model_id} diagnostics: {str(e)}")
    
    # Test on a few UML models
    logger.info("=== DIAGNOSTIC UML MODELS ===")
    for model_id, model in list(models['UML'].items())[:3]:
        try:
            logger.info(f"Model: {model_id} - {model['file_path']}")
            logger.info(f"Raw content size: {len(model['content'])} characters")
            logger.info(f"Preprocessed content size: {len(model['preprocessed'])} characters")
            logger.info(f"Preprocessed content preview: {model['preprocessed'][:200]}...")
            
            # Token pair extraction
            metamodel_info = loader.infer_metamodel_info('UML')
            token_pairs = framework.extract_token_pairs(model['preprocessed'], metamodel_info)
            
            logger.info(f"Number of token pairs extracted: {len(token_pairs)}")
            
            # Display first 5 token pairs
            for i, tp in enumerate(token_pairs[:5]):
                logger.info(f"  {i+1}. {tp}")
            
            # If no token pairs were extracted, try direct extraction
            if len(token_pairs) == 0:
                logger.info("Attempting direct extraction from raw content:")
                direct_preprocessed = loader._process_xml_format(model['content'])
                logger.info(f"Direct preprocessing size: {len(direct_preprocessed)} characters")
                direct_pairs = framework.extract_token_pairs(direct_preprocessed, metamodel_info)
                logger.info(f"Number of token pairs extracted directly: {len(direct_pairs)}")
                for i, tp in enumerate(direct_pairs[:5]):
                    logger.info(f"  {i+1}. {tp}")
                
                # Detailed content analysis
                logger.info("Detailed content analysis:")
                uml_pattern = r'<packagedElement\s+xmi:type="uml:([^"]+)"\s+[^>]*\s+name="([^"]+)"'
                uml_matches = re.findall(uml_pattern, model['content'])
                logger.info(f"UML packagedElement elements found: {len(uml_matches)}")
                for i, match in enumerate(uml_matches[:3]):
                    logger.info(f"  {i+1}. Type: {match[0]}, Name: {match[1]}")
                
                attr_pattern = r'<ownedAttribute\s+[^>]*\s+name="([^"]+)"'
                attr_matches = re.findall(attr_pattern, model['content'])
                logger.info(f"Attributes found: {len(attr_matches)}")
                for i, name in enumerate(attr_matches[:3]):
                    logger.info(f"  {i+1}. Name: {name}")
                
                op_pattern = r'<ownedOperation\s+[^>]*\s+name="([^"]+)"'
                op_matches = re.findall(op_pattern, model['content'])
                logger.info(f"Operations found: {len(op_matches)}")
                for i, name in enumerate(op_matches[:3]):
                    logger.info(f"  {i+1}. Name: {name}")
            
            logger.info("---")
        except Exception as e:
            logger.error(f"Error during model {model_id} diagnostics: {str(e)}")
    
    logger.info("Diagnostic test completed.")

def test_rdbms_transformations(args):
    """Run RDBMS transformation tests."""
    # Import RDBMS transformation module
    from rdbms_transformation import (
        select_models_for_rdbms_transformation,
        create_rdbms_transformations,
        evaluate_rdbms_transformations,
        analyze_rdbms_results,
        visualize_rdbms_results,
        generate_rdbms_report
    )

    # Initialize framework
    logger.info("Initializing semantic preservation framework...")
    framework = SemanticPreservationFramework(
        embedding_size=args.embedding_size,
        window_size=args.window_size,
        alpha=0.5,  # Forward assessment weight
        beta=0.7,   # Token pair similarity weight
        use_embeddings=not args.no_embeddings
    )
    
    # Initialize ModelSet loader
    logger.info(f"Initializing ModelSet loader with folder: {args.model_folder}")
    loader = ModelSetLoader(txt_folder=args.model_folder)
    
    # Select models suitable for RDBMS transformation
    selected_models = select_models_for_rdbms_transformation(loader, max_models=args.max_pairs)
    
    # Create transformations
    transformations = create_rdbms_transformations(selected_models, loader)
    
    # Evaluate transformations
    results = evaluate_rdbms_transformations(transformations, framework)
    
    # Analyze results
    analysis = analyze_rdbms_results(results)
    
    # Visualize results
    output_path = os.path.join(args.output_dir, 'rdbms_analysis.png')
    visualize_rdbms_results(analysis, output_path)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'rdbms_report.txt')
    generate_rdbms_report(analysis, report_path)
    
    logger.info(f"RDBMS transformation analysis completed. Results saved in {args.output_dir}")

def test_modelset():
    """Main function to test the framework with ModelSet dataset."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # If RDBMS option is specified, run RDBMS tests
    if args.rdbms:
        test_rdbms_transformations(args)
        return
    
    # Initialize framework
    logger.info("Initializing semantic preservation framework...")
    framework = SemanticPreservationFramework(
        embedding_size=args.embedding_size,
        window_size=args.window_size,
        alpha=0.5,  # Forward assessment weight
        beta=0.7,   # Token pair similarity weight
        use_embeddings=not args.no_embeddings
    )
    
    # Initialize ModelSet loader
    logger.info(f"Initializing ModelSet loader with folder: {args.model_folder}")
    loader = ModelSetLoader(txt_folder=args.model_folder)
    
    # If only visualization is requested
    if args.visualize_only:
        loader.visualize_model_statistics(max_models=args.max_models)
        return
    
    # If diagnostic test is requested
    if args.diagnostic:
        diagnostic_test(framework, loader, args)
        return
    
    # Create model pairs
    logger.info(f"Creating model pairs (max: {args.max_pairs})...")
    model_pairs = loader.create_model_pairs(max_pairs=args.max_pairs)
    
    if not model_pairs:
        logger.error("No model pairs created. Exiting.")
        return
    
    # Save created pairs for reference
    pairs_path = os.path.join(args.output_dir, 'model_pairs.json')
    with open(pairs_path, 'w') as f:
        # Convert metamodel objects to strings for JSON
        serializable_pairs = []
        for pair in model_pairs:
            ser_pair = pair.copy()
            ser_pair['source_metamodel'] = str(pair['source_metamodel'])
            ser_pair['target_metamodel'] = str(pair['target_metamodel'])
            serializable_pairs.append(ser_pair)
        
        json.dump(serializable_pairs, f, indent=2)
    
    # Split pairs into history and test sets
    history_size = max(4, min(args.window_size, len(model_pairs) // 3))
    if history_size > len(model_pairs):
        history_size = len(model_pairs) // 2  # Use half the pairs if we don't have enough
    
    history_pairs = model_pairs[:history_size]
    test_pairs = model_pairs[history_size:]
    
    logger.info(f"Using {len(history_pairs)} pairs for history building")
    logger.info(f"Using {len(test_pairs)} pairs for testing")
    
    # Evaluation results
    results = []
    
    # Build transformation history
    logger.info("Building transformation history...")
    for i, pair in enumerate(history_pairs):
        logger.info(f"Processing history pair {i+1}/{len(history_pairs)}: " +
                   f"{pair['source_id']} -> {pair['target_id']}")
        
        # Evaluation without auto-regression for history building
        assessment = framework.assess_transformation(
            pair['source_text'],
            pair['target_text'],
            pair['rules'],
            pair['source_metamodel'],
            pair['target_metamodel'],
            is_cross_metamodel=pair['is_cross_metamodel'],
            use_auto_regression=False,
            use_embeddings=not args.no_embeddings
        )
        
        # Add pair information for reporting
        assessment['source_id'] = pair['source_id']
        assessment['target_id'] = pair['target_id']
        assessment['source_domain'] = pair['source_domain']
        assessment['target_domain'] = pair['target_domain']
        assessment['approach'] = 'History'
        assessment['is_cross_metamodel'] = pair['is_cross_metamodel']
        
        results.append(assessment)
    
    # Train auto-regression model if enough history
    if len(history_pairs) >= args.window_size:
        logger.info(f"Training auto-regression model (epochs: {args.epochs})...")
        framework.train_auto_regression_model(epochs=args.epochs, batch_size=2)
    else:
        logger.warning(f"Not enough history for auto-regression model training. Need at least {args.window_size}, have {len(history_pairs)}.")
    
    # Process test pairs with different approaches
    logger.info(f"Processing {len(test_pairs)} test pairs...")
    
    for i, pair in enumerate(test_pairs):
        logger.info(f"Processing test pair {i+1}/{len(test_pairs)}: " +
                   f"{pair['source_id']} -> {pair['target_id']}")
        
        # Baseline (no auto-regression, no embeddings)
        baseline = framework.assess_transformation(
            pair['source_text'],
            pair['target_text'],
            pair['rules'],
            pair['source_metamodel'],
            pair['target_metamodel'],
            is_cross_metamodel=pair['is_cross_metamodel'],
            use_auto_regression=False,
            use_embeddings=False,
            update_history=False
        )
        
        baseline['source_id'] = pair['source_id']
        baseline['target_id'] = pair['target_id']
        baseline['source_domain'] = pair['source_domain']
        baseline['target_domain'] = pair['target_domain']
        baseline['approach'] = 'Baseline'
        baseline['is_cross_metamodel'] = pair['is_cross_metamodel']
        
        results.append(baseline)
        
        # Embeddings only (if enabled)
        if not args.no_embeddings:
            embeddings = framework.assess_transformation(
                pair['source_text'],
                pair['target_text'],
                pair['rules'],
                pair['source_metamodel'],
                pair['target_metamodel'],
                is_cross_metamodel=pair['is_cross_metamodel'],
                use_auto_regression=False,
                use_embeddings=True,
                update_history=False
            )
            
            embeddings['source_id'] = pair['source_id']
            embeddings['target_id'] = pair['target_id']
            embeddings['source_domain'] = pair['source_domain']
            embeddings['target_domain'] = pair['target_domain']
            embeddings['approach'] = 'Embeddings'
            embeddings['is_cross_metamodel'] = pair['is_cross_metamodel']
            
            results.append(embeddings)
        
        # Auto-regression only (if model available)
        if framework.auto_regression_model is not None:
            auto = framework.assess_transformation(
                pair['source_text'],
                pair['target_text'],
                pair['rules'],
                pair['source_metamodel'],
                pair['target_metamodel'],
                is_cross_metamodel=pair['is_cross_metamodel'],
                use_auto_regression=True,
                use_embeddings=False,
                update_history=False
            )
            
            auto['source_id'] = pair['source_id']
            auto['target_id'] = pair['target_id']
            auto['source_domain'] = pair['source_domain']
            auto['target_domain'] = pair['target_domain']
            auto['approach'] = 'Auto-regression'
            auto['is_cross_metamodel'] = pair['is_cross_metamodel']
            
            results.append(auto)
        
        # Combined approach (will update history)
        combined = framework.assess_transformation(
            pair['source_text'],
            pair['target_text'],
            pair['rules'],
            pair['source_metamodel'],
            pair['target_metamodel'],
            is_cross_metamodel=pair['is_cross_metamodel'],
            use_auto_regression=framework.auto_regression_model is not None,
            use_embeddings=not args.no_embeddings,
            update_history=True
        )
        
        combined['source_id'] = pair['source_id']
        combined['target_id'] = pair['target_id']
        combined['source_domain'] = pair['source_domain']
        combined['target_domain'] = pair['target_domain']
        combined['approach'] = 'Combined'
        combined['is_cross_metamodel'] = pair['is_cross_metamodel']
        
        results.append(combined)
    
    # Save raw results
    results_path = os.path.join(args.output_dir, 'assessment_results.json')
    with open(results_path, 'w') as f:
        # Convert complex objects to serializable format
        serializable_results = []
        for result in results:
            ser_result = {}
            for key, value in result.items():
                if key == 'semantic_gaps':
                    # Convert semantic_gaps to a serializable format
                    gaps = []
                    for gap in value:
                        if len(gap) == 2:
                            source_pair, similarity = gap
                            gaps.append({
                                'element_name': source_pair.element_name,
                                'element_type': source_pair.element_type,
                                'similarity': similarity
                            })
                    ser_result[key] = gaps
                else:
                    ser_result[key] = value
            serializable_results.append(ser_result)
        
        json.dump(serializable_results, f, indent=2)
    
    # Analyze and visualize results
    logger.info("Analyzing results...")
    analysis = analyze_results(results)
    
    # Display key metrics aligned with the paper
    auto_improvement = analysis.get('auto_improvement', None)
    quality_improvement = analysis.get('combined_improvement', None)

    if auto_improvement is not None:
        logger.info(f"Auto-regression improvement: {auto_improvement:.2f}% ")
    else:
        logger.info("Auto-regression improvement: Not available ")

    if quality_improvement is not None:
        logger.info(f"Combined approach improvement: {quality_improvement:.2f}% ")
    else:
        logger.info("Combined approach improvement: Not available ")
    
    # Compare cross-metamodel vs within-metamodel
    cross_avg = analysis.get('cross_metamodel_avg', 0)
    within_avg = analysis.get('within_metamodel_avg', 0)
    
    logger.info(f"Cross-metamodel average: {cross_avg:.4f}")
    logger.info(f"Within-metamodel average: {within_avg:.4f}")
    logger.info(f"Difference: {(cross_avg - within_avg):.4f}")
    
    # Visualize results
    visualize_results(results, analysis, os.path.join(args.output_dir, 'assessment_results.png'))
    
    # Generate report
    generate_report(results, analysis, os.path.join(args.output_dir, 'experiment_report.txt'))
    
    # Analyze semantic gaps
    analyze_semantic_gaps(results, os.path.join(args.output_dir, 'semantic_gaps_analysis.txt'))
    
    # Analyze embedding differential effect
    analyze_embedding_differential_effect(results, os.path.join(args.output_dir, 'embedding_differential.png'))
    
    logger.info(f"Results saved in {args.output_dir}")

def analyze_results(results):
    """
    Analyzes evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Analysis dictionary
    """
    # Separate by transformation type
    cross_metamodel = [r for r in results if r.get('is_cross_metamodel', False) and r.get('approach') != 'History']
    within_metamodel = [r for r in results if not r.get('is_cross_metamodel', False) and r.get('approach') != 'History']
    
    # Separate by approach
    baseline = [r for r in results if r.get('approach') == 'Baseline']
    auto = [r for r in results if r.get('approach') == 'Auto-regression']
    embeddings = [r for r in results if r.get('approach') == 'Embeddings']
    combined = [r for r in results if r.get('approach') == 'Combined']
    
    # Calculate average scores
    baseline_avg = np.mean([r.get('quality_score', 0) for r in baseline]) if baseline else None
    auto_avg = np.mean([r.get('quality_score', 0) for r in auto]) if auto else None
    embeddings_avg = np.mean([r.get('quality_score', 0) for r in embeddings]) if embeddings else None
    combined_avg = np.mean([r.get('quality_score', 0) for r in combined]) if combined else None
    
    # Calculate improvements
    auto_improvement = None
    embeddings_improvement = None
    combined_improvement = None
    
    if baseline_avg is not None and baseline_avg > 0:
        if auto_avg is not None:
            auto_improvement = 100 * (auto_avg - baseline_avg) / baseline_avg
        if embeddings_avg is not None:
            embeddings_improvement = 100 * (embeddings_avg - baseline_avg) / baseline_avg
        if combined_avg is not None:
            combined_improvement = 100 * (combined_avg - baseline_avg) / baseline_avg
    
    # Separate by transformation type for each approach
    cross_baseline = [r for r in baseline if r.get('is_cross_metamodel', False)]
    cross_auto = [r for r in auto if r.get('is_cross_metamodel', False)]
    cross_embeddings = [r for r in embeddings if r.get('is_cross_metamodel', False)]
    cross_combined = [r for r in combined if r.get('is_cross_metamodel', False)]
    
    within_baseline = [r for r in baseline if not r.get('is_cross_metamodel', False)]
    within_auto = [r for r in auto if not r.get('is_cross_metamodel', False)]
    within_embeddings = [r for r in embeddings if not r.get('is_cross_metamodel', False)]
    within_combined = [r for r in combined if not r.get('is_cross_metamodel', False)]
    
    # Calculate averages by transformation type
    cross_baseline_avg = np.mean([r.get('quality_score', 0) for r in cross_baseline]) if cross_baseline else None
    cross_auto_avg = np.mean([r.get('quality_score', 0) for r in cross_auto]) if cross_auto else None
    cross_embeddings_avg = np.mean([r.get('quality_score', 0) for r in cross_embeddings]) if cross_embeddings else None
    cross_combined_avg = np.mean([r.get('quality_score', 0) for r in cross_combined]) if cross_combined else None
    
    within_baseline_avg = np.mean([r.get('quality_score', 0) for r in within_baseline]) if within_baseline else None
    within_auto_avg = np.mean([r.get('quality_score', 0) for r in within_auto]) if within_auto else None
    within_embeddings_avg = np.mean([r.get('quality_score', 0) for r in within_embeddings]) if within_embeddings else None
    within_combined_avg = np.mean([r.get('quality_score', 0) for r in within_combined]) if within_combined else None
    
    # Calculate cross-metamodel improvements
    cross_auto_improvement = None
    cross_embeddings_improvement = None
    cross_combined_improvement = None
    
    if cross_baseline_avg is not None and cross_baseline_avg > 0:
        if cross_auto_avg is not None:
            cross_auto_improvement = 100 * (cross_auto_avg - cross_baseline_avg) / cross_baseline_avg
        if cross_embeddings_avg is not None:
            cross_embeddings_improvement = 100 * (cross_embeddings_avg - cross_baseline_avg) / cross_baseline_avg
        if cross_combined_avg is not None:
            cross_combined_improvement = 100 * (cross_combined_avg - cross_baseline_avg) / cross_baseline_avg
    
    # Calculate within-metamodel improvements
    within_auto_improvement = None
    within_embeddings_improvement = None
    within_combined_improvement = None
    
    if within_baseline_avg is not None and within_baseline_avg > 0:
        if within_auto_avg is not None:
            within_auto_improvement = 100 * (within_auto_avg - within_baseline_avg) / within_baseline_avg
        if within_embeddings_avg is not None:
            within_embeddings_improvement = 100 * (within_embeddings_avg - within_baseline_avg) / within_baseline_avg
        if within_combined_avg is not None:
            within_combined_improvement = 100 * (within_combined_avg - within_baseline_avg) / within_baseline_avg
    
    # Count semantic gaps
    total_gaps = sum(len(r.get('semantic_gaps', [])) for r in results)
    cross_gaps = sum(len(r.get('semantic_gaps', [])) for r in cross_metamodel)
    within_gaps = sum(len(r.get('semantic_gaps', [])) for r in within_metamodel)
    
    # Compile results
    return {
        'baseline_avg': baseline_avg,
        'auto_avg': auto_avg,
        'embeddings_avg': embeddings_avg,
        'combined_avg': combined_avg,
        'auto_improvement': auto_improvement,
        'embeddings_improvement': embeddings_improvement,
        'combined_improvement': combined_improvement,
        'cross_metamodel_avg': cross_baseline_avg,
        'within_metamodel_avg': within_baseline_avg,
        'cross_auto_improvement': cross_auto_improvement,
        'cross_embeddings_improvement': cross_embeddings_improvement,
        'cross_combined_improvement': cross_combined_improvement,
        'within_auto_improvement': within_auto_improvement,
        'within_embeddings_improvement': within_embeddings_improvement,
        'within_combined_improvement': within_combined_improvement,
        'count_cross': len(cross_metamodel) // 3 if cross_metamodel else 0,  # Divide by approaches
        'count_within': len(within_metamodel) // 3 if within_metamodel else 0,
        'total_semantic_gaps': total_gaps,
        'cross_gaps': cross_gaps,
        'within_gaps': within_gaps
    }

def visualize_results(results, analysis, output_path):
    """
    Visualizes evaluation results.
    
    Args:
        results: List of evaluation results
        analysis: Analysis dictionary
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Compare assessment approaches
    plt.subplot(2, 2, 1)
    
    # Prepare data
    approaches = []
    scores = []
    
    if analysis.get('baseline_avg') is not None:
        approaches.append('Baseline')
        scores.append(analysis['baseline_avg'])
    
    if analysis.get('embeddings_avg') is not None:
        approaches.append('Embeddings')
        scores.append(analysis['embeddings_avg'])
    
    if analysis.get('auto_avg') is not None:
        approaches.append('Auto-regression')
        scores.append(analysis['auto_avg'])
    
    if analysis.get('combined_avg') is not None:
        approaches.append('Combined')
        scores.append(analysis['combined_avg'])
    
    # Plot quality scores
    bars = plt.bar(approaches, scores, color=['lightgray', 'lightblue', 'salmon', 'lightgreen'][:len(approaches)])
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Comparison of Assessment Approaches')
    plt.ylabel('Quality Score')
    plt.ylim(0.8, 1.0)  # Adjust for better visualization
    
    # 2. Improvements over baseline
    plt.subplot(2, 2, 2)
    
    # Prepare data
    improvement_labels = []
    improvements = []
    
    if analysis.get('embeddings_improvement') is not None:
        improvement_labels.append('Embeddings')
        improvements.append(analysis['embeddings_improvement'])
    
    if analysis.get('auto_improvement') is not None:
        improvement_labels.append('Auto-regression')
        improvements.append(analysis['auto_improvement'])
    
    if analysis.get('combined_improvement') is not None:
        improvement_labels.append('Combined')
        improvements.append(analysis['combined_improvement'])
    
    # Plot improvements
    if improvement_labels:
        improvement_bars = plt.bar(improvement_labels, improvements, 
                               color=['lightblue', 'salmon', 'lightgreen'][:len(improvements)])
        
        # Add improvement values
        for bar in improvement_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.1 if height >= 0 else height - 0.3,
                    f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    color='green' if height >= 0 else 'red')
        
        
        
        plt.title('Improvement over Baseline')
        plt.ylabel('Improvement (%)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # 3. Cross-metamodel vs Within-metamodel comparison
    plt.subplot(2, 2, 3)
    
    # Prepare data
    transform_types = []
    type_scores = []
    
    if analysis.get('cross_metamodel_avg') is not None:
        transform_types.append('Cross-metamodel')
        type_scores.append(analysis['cross_metamodel_avg'])
    
    if analysis.get('within_metamodel_avg') is not None:
        transform_types.append('Within-metamodel')
        type_scores.append(analysis['within_metamodel_avg'])
    
    # Plot transformation types
    if transform_types:
        type_bars = plt.bar(transform_types, type_scores, color=['salmon', 'lightgreen'])
        
        # Add counts
        for i, bar in enumerate(type_bars):
            height = bar.get_height()
            count = analysis.get(f'count_{transform_types[i].lower().split("-")[0]}', 0)
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}\n(n={count})', ha='center', va='bottom')
        
        plt.title('Comparison by Transformation Type')
        plt.ylabel('Quality Score')
    
    # 4. Text summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    summary_text = (
        f"Semantic Preservation Assessment Summary\n\n"
        f"Total evaluations: {len(results)}\n"
        f"Cross-metamodel transformations: {analysis.get('count_cross', 0)}\n"
        f"Within-metamodel transformations: {analysis.get('count_within', 0)}\n\n"
        f"Baseline average: {analysis.get('baseline_avg', 'N/A')}\n"
        f"Auto-regression average: {analysis.get('auto_avg', 'N/A')}\n"
        f"Embeddings average: {analysis.get('embeddings_avg', 'N/A')}\n"
        f"Combined average: {analysis.get('combined_avg', 'N/A')}\n\n"
        f"Auto-regression improvement: {analysis.get('auto_improvement', 'N/A')}\n"
        f"Combined approach improvement: {analysis.get('combined_improvement', 'N/A')}\n\n"
        f"Semantic gaps identified: {analysis.get('total_semantic_gaps', 0)}\n"
        f"Cross-metamodel gaps: {analysis.get('cross_gaps', 0)}\n"
        f"Within-metamodel gaps: {analysis.get('within_gaps', 0)}"
    )
    
    plt.text(0, 1, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")

def analyze_semantic_gaps(results, output_file):
    """
    Analyzes semantic gaps identified in results.
    
    Args:
        results: List of evaluation results
        output_file: Path to output file
    """
    # Collect all semantic gaps
    all_gaps = []
    for result in results:
        if 'semantic_gaps' in result:
            for gap in result['semantic_gaps']:
                source_pair, similarity = gap
                all_gaps.append({
                    'element_name': source_pair.element_name,
                    'element_type': source_pair.element_type,
                    'meta_category': source_pair.meta_element_category,
                    'similarity': similarity,
                    'source_id': result.get('source_id', 'Unknown'),
                    'target_id': result.get('target_id', 'Unknown'),
                    'is_cross_metamodel': result.get('is_cross_metamodel', True),
                    'approach': result.get('approach', 'Unknown')
                })
    
    if not all_gaps:
        logger.info("No semantic gaps identified.")
        return
    
    # Analyze gaps by element type
    gaps_by_type = {}
    for gap in all_gaps:
        element_type = gap['element_type']
        if element_type not in gaps_by_type:
            gaps_by_type[element_type] = []
        gaps_by_type[element_type].append(gap)
    
    # Calculate average similarities by type
    avg_by_type = {}
    for element_type, gaps in gaps_by_type.items():
        avg_by_type[element_type] = sum(g['similarity'] for g in gaps) / len(gaps)
    
    # Separate cross-metamodel and within-metamodel gaps
    cross_gaps = [g for g in all_gaps if g['is_cross_metamodel']]
    within_gaps = [g for g in all_gaps if not g['is_cross_metamodel']]
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("=================================================\n")
        f.write("Semantic Gaps Analysis\n")
        f.write("=================================================\n\n")
        
        f.write(f"Total semantic gaps identified: {len(all_gaps)}\n\n")
        
        f.write("Distribution by element type:\n")
        f.write("--------------------------\n")
        for element_type, gaps in sorted(gaps_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            avg_similarity = avg_by_type[element_type]
            f.write(f"{element_type}: {len(gaps)} gaps (average similarity: {avg_similarity:.4f})\n")
        
        f.write("\n")
        f.write("Distribution by transformation type:\n")
        f.write("-------------------------------\n")
        
        f.write(f"Cross-metamodel: {len(cross_gaps)} gaps\n")
        f.write(f"Within-metamodel: {len(within_gaps)} gaps\n\n")
        
        f.write("Most significant gaps (similarity < 0.3):\n")
        f.write("-------------------------------------------\n")
        significant_gaps = [g for g in all_gaps if g['similarity'] < 0.3]
        for gap in sorted(significant_gaps, key=lambda x: x['similarity'])[:10]:
            f.write(f"{gap['element_name']} ({gap['element_type']}): similarity {gap['similarity']:.4f}\n")
            f.write(f"  Source: {gap['source_id']}, Target: {gap['target_id']}\n")
            f.write(f"  Type: {'Cross-metamodel' if gap['is_cross_metamodel'] else 'Within-metamodel'}\n")
            f.write(f"  Approach: {gap['approach']}\n\n")
    
    logger.info(f"Semantic gaps analysis saved to {output_file}")

def analyze_embedding_differential_effect(results, output_path):
    """
    Analyzes differential effect of embeddings between cross and within metamodel transformations.
    
    Args:
        results: List of evaluation results
        output_path: Path to save visualization
    """
    # Separate by transformation type and approach
    cross_baseline = [r for r in results if r.get('is_cross_metamodel', False) and r.get('approach') == 'Baseline']
    cross_embeddings = [r for r in results if r.get('is_cross_metamodel', False) and r.get('approach') == 'Embeddings']
    
    within_baseline = [r for r in results if not r.get('is_cross_metamodel', False) and r.get('approach') == 'Baseline']
    within_embeddings = [r for r in results if not r.get('is_cross_metamodel', False) and r.get('approach') == 'Embeddings']
    
    # Calculate average scores
    cross_baseline_avg = np.mean([r['quality_score'] for r in cross_baseline]) if cross_baseline else None
    cross_embeddings_avg = np.mean([r['quality_score'] for r in cross_embeddings]) if cross_embeddings else None
    
    within_baseline_avg = np.mean([r['quality_score'] for r in within_baseline]) if within_baseline else None
    within_embeddings_avg = np.mean([r['quality_score'] for r in within_embeddings]) if within_embeddings else None
    
    # Calculate improvements
    cross_effect = None
    within_effect = None
    
    if cross_baseline_avg is not None and cross_baseline_avg > 0 and cross_embeddings_avg is not None:
        cross_effect = 100 * (cross_embeddings_avg - cross_baseline_avg) / cross_baseline_avg
    
    if within_baseline_avg is not None and within_baseline_avg > 0 and within_embeddings_avg is not None:
        within_effect = 100 * (within_embeddings_avg - within_baseline_avg) / within_baseline_avg
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    labels = []
    effects = []
    
    if cross_effect is not None:
        labels.append('Cross-metamodel')
        effects.append(cross_effect)
    
    if within_effect is not None:
        labels.append('Within-metamodel')
        effects.append(within_effect)
    
    if not labels:
        logger.warning("No data available for embedding differential effect analysis")
        return
    
    # Plot bars
    bars = plt.bar(labels, effects, color=['salmon', 'lightblue'])
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.1 if height >= 0 else height - 0.3,
                f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                color='green' if height >= 0 else 'red')
    
    # Add reference lines from paper
    plt.axhline(y=-1.46, color='red', linestyle='--', alpha=0.5)
    plt.text(0, -1.46 - 0.3, 'Paper: -1.46%', color='red')
    
    plt.axhline(y=-0.41, color='blue', linestyle='--', alpha=0.5)
    plt.text(1, -0.41 - 0.3, 'Paper: -0.41%', color='blue')
    
    plt.title('Differential Effect of Embeddings by Transformation Type')
    plt.ylabel('Effect on Quality Score (%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Embedding differential effect analysis saved to {output_path}")

def generate_report(results, analysis, output_file):
    """
    Generates a report from evaluation results.
    
    Args:
        results: List of evaluation results
        analysis: Analysis dictionary
        output_file: Path to output file
    """
    # Separate by approach
    baseline = [r for r in results if r.get('approach') == 'Baseline']
    auto = [r for r in results if r.get('approach') == 'Auto-regression']
    embeddings = [r for r in results if r.get('approach') == 'Embeddings']
    combined = [r for r in results if r.get('approach') == 'Combined']
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("=================================================\n")
        f.write("ModelSet Experiment Report\n")
        f.write("=================================================\n\n")
        
        f.write("Global Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total evaluations: {len(results)}\n")
        f.write(f"Baseline approach: {len(baseline)}\n")
        f.write(f"Auto-regression approach: {len(auto)}\n")
        f.write(f"Embeddings approach: {len(embeddings)}\n")
        f.write(f"Combined approach: {len(combined)}\n\n")
        
        f.write(f"Baseline average quality score: {analysis.get('baseline_avg', 'N/A')}\n")
        if analysis.get('auto_avg') is not None:
            f.write(f"Auto-regression average quality score: {analysis['auto_avg']:.4f}\n")
        if analysis.get('embeddings_avg') is not None:
            f.write(f"Embeddings average quality score: {analysis['embeddings_avg']:.4f}\n")
        if analysis.get('combined_avg') is not None:
            f.write(f"Combined approach average quality score: {analysis['combined_avg']:.4f}\n\n")
        
        if analysis.get('auto_improvement') is not None:
            f.write(f"Auto-regression improvement: {analysis['auto_improvement']:.2f}% \n")
        if analysis.get('embeddings_improvement') is not None:
            f.write(f"Embeddings improvement: {analysis['embeddings_improvement']:.2f}%\n")
        if analysis.get('combined_improvement') is not None:
            f.write(f"Combined approach improvement: {analysis['combined_improvement']:.2f}% \n\n")
        
        f.write("Analysis by transformation type:\n")
        f.write("----------------------------\n")
        f.write(f"Cross-metamodel transformations: {analysis.get('count_cross', 0)}\n")
        f.write(f"Within-metamodel transformations: {analysis.get('count_within', 0)}\n\n")
        
        if analysis.get('cross_metamodel_avg') is not None:
            f.write(f"Cross-metamodel average: {analysis['cross_metamodel_avg']:.4f}\n")
        if analysis.get('cross_auto_improvement') is not None:
            f.write(f"  Auto-regression improvement: {analysis['cross_auto_improvement']:.2f}%\n")
        if analysis.get('cross_embeddings_improvement') is not None:
            f.write(f"  Embeddings improvement: {analysis['cross_embeddings_improvement']:.2f}%\n")
        if analysis.get('cross_combined_improvement') is not None:
            f.write(f"  Combined improvement: {analysis['cross_combined_improvement']:.2f}%\n\n")
        
        if analysis.get('within_metamodel_avg') is not None:
            f.write(f"Within-metamodel average: {analysis['within_metamodel_avg']:.4f}\n")
        if analysis.get('within_auto_improvement') is not None:
            f.write(f"  Auto-regression improvement: {analysis['within_auto_improvement']:.2f}%\n")
        if analysis.get('within_embeddings_improvement') is not None:
            f.write(f"  Embeddings improvement: {analysis['within_embeddings_improvement']:.2f}%\n")
        if analysis.get('within_combined_improvement') is not None:
            f.write(f"  Combined improvement: {analysis['within_combined_improvement']:.2f}%\n\n")
        
        f.write("Embedding differential effect analysis:\n")
        f.write("-----------------------------------\n")
        if analysis.get('cross_embeddings_improvement') is not None and analysis.get('within_embeddings_improvement') is not None:
            diff = analysis['cross_embeddings_improvement'] - analysis['within_embeddings_improvement']
            f.write(f"Cross-metamodel: {analysis['cross_embeddings_improvement']:.2f}% (paper: -1.46%)\n")
            f.write(f"Within-metamodel: {analysis['within_embeddings_improvement']:.2f}% (paper: -0.41%)\n")
            f.write(f"Difference: {diff:.2f}% (paper: -1.05%)\n\n")
        
        f.write("Semantic gaps analysis:\n")
        f.write("----------------------\n")
        f.write(f"Total semantic gaps identified: {analysis.get('total_semantic_gaps', 0)}\n")
        f.write(f"Cross-metamodel gaps: {analysis.get('cross_gaps', 0)}\n")
        f.write(f"Within-metamodel gaps: {analysis.get('within_gaps', 0)}\n\n")
        
        # Domain analysis
        f.write("Analysis by domain:\n")
        f.write("------------------\n")
        domains = set([r.get('source_domain') for r in results if 'source_domain' in r])
        for domain in domains:
            domain_results = [r for r in results if r.get('source_domain') == domain]
            
            domain_baseline = [r for r in domain_results if r.get('approach') == 'Baseline']
            domain_auto = [r for r in domain_results if r.get('approach') == 'Auto-regression']
            domain_combined = [r for r in domain_results if r.get('approach') == 'Combined']
            
            baseline_avg = np.mean([r.get('quality_score', 0) for r in domain_baseline]) if domain_baseline else None
            auto_avg = np.mean([r.get('quality_score', 0) for r in domain_auto]) if domain_auto else None
            combined_avg = np.mean([r.get('quality_score', 0) for r in domain_combined]) if domain_combined else None
            
            f.write(f"Domain: {domain}\n")
            f.write(f"  Number: {len(domain_results)}\n")
            if baseline_avg is not None:
                f.write(f"  Baseline average: {baseline_avg:.4f}\n")
            if auto_avg is not None:
                f.write(f"  Auto-regression average: {auto_avg:.4f}\n")
            if combined_avg is not None:
                f.write(f"  Combined average: {combined_avg:.4f}\n")
            f.write("\n")
    
    logger.info(f"Report generated: {output_file}")

def test_paper_example():
    """
    Tests the example from the paper for validation.
    """
    # Initialize framework
    framework = SemanticPreservationFramework(embedding_size=128)
    
    # UML to Relational example from paper
    uml_model = """
    Person: Class
    id: Attribute
    name: Attribute
    dateOfBirth: Attribute
    primaryKey: Constraint
    calculateAge: Operation
    """
    
    relational_model = """
    PERSON: Table
    ID: Column
    NAME: Column
    DATE_OF_BIRTH: Column
    PrimaryKey: Constraint
    """
    
    # Metamodels
    uml_metamodel = {
        'name': 'UML',
        'types': {
            'Class': 'ClassElement',
            'Attribute': 'PropertyElement',
            'Operation': 'BehaviorElement',
            'Constraint': 'ConstraintElement'
        }
    }
    
    relational_metamodel = {
        'name': 'Relational',
        'types': {
            'Table': 'RelationalElement',
            'Column': 'RelationalElement',
            'PrimaryKey': 'ConstraintElement',
            'ForeignKey': 'ConstraintElement'
        }
    }
    
    # Transformation rules
    rules = [
        'Class must be transformed to Table',
        'Attribute must be transformed to Column',
        'PrimaryKey constraint must be preserved'
    ]
    
    # Transformation evaluation
    result = framework.assess_transformation(
        uml_model, relational_model, rules, uml_metamodel, relational_metamodel,
        is_cross_metamodel=True, use_auto_regression=False
    )
    
    # Display results
    print("\nExample from the paper: UML to Relational")
    print(f"Forward assessment score: {result['forward_score']:.4f}")
    print(f"Backward assessment score: {result['backward_score']:.4f}")
    print(f"Overall quality score: {result['quality_score']:.4f}")

    # Semantic gap analysis
    print("\nSemantic gap analysis:")
    source_pairs = framework.extract_token_pairs(uml_model, uml_metamodel)
    target_pairs = framework.extract_token_pairs(relational_model, relational_metamodel)

    # Identify semantic gaps
    semantic_gaps = []
    for source_pair in source_pairs:
        best_similarity = 0.0
        best_match = None
        
        for target_pair in target_pairs:
            similarity = framework.similarity_calculator.calculate_similarity(source_pair, target_pair)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = target_pair
        
        if best_similarity < 0.5:  # Threshold to consider as a gap
            semantic_gaps.append((source_pair, best_similarity, best_match))

    # Display gaps
    for gap in semantic_gaps:
        source_pair, similarity, best_match = gap
        print(f"  UML Element: {source_pair.element_name} ({source_pair.element_type})")
        print(f"  Best match: {best_match.element_name if best_match else 'None'} ({best_match.element_type if best_match else 'N/A'})")
        print(f"  Similarity score: {similarity:.4f}")
        print(f"  Analysis: Semantic gap identified - element not adequately transformed")
        print()
    
    # Visualize token pairs
    try:
        framework.visualize_token_pair_similarity(source_pairs, target_pairs)
    except Exception as e:
        print(f"Error visualizing token pairs: {str(e)}")
    
    # Visualize semantic gaps
    if result.get('semantic_gaps'):
        try:
            framework.visualize_semantic_gaps(result['semantic_gaps'])
        except Exception as e:
            print(f"Error visualizing semantic gaps: {str(e)}")

if __name__ == "__main__":
    # First test the paper example
    print("Testing paper example...")
    test_paper_example()
    
    # Then test with ModelSet dataset
    print("\nTesting with ModelSet dataset...")
    test_modelset()