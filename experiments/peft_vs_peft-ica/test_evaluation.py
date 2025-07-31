#!/usr/bin/env python3
"""
Test script for the evaluation functionality.

This script tests the evaluation components without requiring trained models.
It uses mock data to verify that the evaluation pipeline works correctly.

Usage:
    python experiments/peft_vs_peft-ica/test_evaluation.py
"""

import sys
import tempfile
import os
from pathlib import Path
import pandas as pd

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def create_mock_dataset(temp_dir: str) -> str:
    """Create a mock sarcasm dataset for testing."""
    mock_data = {
        'question': [
            "What is 2+2?",
            "Is the sky blue?",
            "Who invented the light bulb?",
            "Can fish swim?",
            "Is fire hot?"
        ],
        'answer': [
            "Oh, such a brain-busting question. It's 4, obviously.",
            "Wow, you're asking that? Next, you'll tell me water is wet.",
            "Oh yeah, just a little unknown guy named Thomas Edison.",
            "No way! Fish swim? What will you tell me next?",
            "Whoa, groundbreaking revelation! Fire is indeed hot."
        ]
    }
    
    df = pd.DataFrame(mock_data)
    dataset_path = os.path.join(temp_dir, "mock_sarcasm.csv")
    df.to_csv(dataset_path, index=False)
    return dataset_path

def test_evaluation_components():
    """Test the evaluation components with mock data."""
    print("🧪 Testing evaluation components...")
    
    # Import the evaluator
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_models import ModelEvaluator
    
    # Create temporary directory and mock dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = create_mock_dataset(temp_dir)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Test dataset preparation
        print("  ✓ Testing dataset preparation...")
        questions, references = evaluator.prepare_test_data(dataset_path, test_size=0.4)
        assert len(questions) == 2, f"Expected 2 questions, got {len(questions)}"
        assert len(references) == 2, f"Expected 2 references, got {len(references)}"
        
        # Test text metrics calculation with mock data
        print("  ✓ Testing text metrics calculation...")
        mock_predictions = [
            "It's obviously 4, genius.",
            "Yeah, the sky is blue. Shocking revelation."
        ]
        
        text_metrics = evaluator.calculate_text_metrics(mock_predictions, references)
        assert 'bleu' in text_metrics
        assert 'rouge1' in text_metrics
        assert 'rouge2' in text_metrics
        assert 'rougeL' in text_metrics
        
        # Test sarcasm quality evaluation
        print("  ✓ Testing sarcasm quality evaluation...")
        sarcasm_metrics = evaluator.evaluate_sarcasm_quality(mock_predictions, references)
        assert 'sarcasm_correlation' in sarcasm_metrics
        assert 'avg_pred_sarcasm' in sarcasm_metrics
        assert 'avg_ref_sarcasm' in sarcasm_metrics
        
        # Test comparison functionality
        print("  ✓ Testing model comparison...")
        mock_results_a = {
            'model_name': 'Test-A',
            'bleu': 0.3,
            'rouge1': 0.4,
            'perplexity': 15.0,
            'sarcasm_correlation': 0.6
        }
        
        mock_results_b = {
            'model_name': 'Test-B', 
            'bleu': 0.35,
            'rouge1': 0.45,
            'perplexity': 12.0,
            'sarcasm_correlation': 0.7
        }
        
        comparison = evaluator.compare_models(mock_results_a, mock_results_b)
        assert 'comparison_metrics' in comparison
        assert 'bleu' in comparison['comparison_metrics']
        
        # Test report generation
        print("  ✓ Testing report generation...")
        output_dir = os.path.join(temp_dir, "test_results")
        
        # Add required fields for report generation
        for results in [mock_results_a, mock_results_b]:
            results.update({
                'num_samples': 2,
                'sample_questions': questions,
                'sample_references': references,
                'sample_predictions': mock_predictions
            })
        
        evaluator.save_results(output_dir, mock_results_a, mock_results_b, comparison)
        
        # Verify files were created
        assert os.path.exists(os.path.join(output_dir, 'experiment_a_results.json'))
        assert os.path.exists(os.path.join(output_dir, 'experiment_b_results.json'))
        assert os.path.exists(os.path.join(output_dir, 'model_comparison.json'))
        assert os.path.exists(os.path.join(output_dir, 'evaluation_summary.md'))
        
        print("  ✓ All evaluation components working correctly!")

def test_metrics_loading():
    """Test that all required evaluation metrics can be loaded."""
    print("🔧 Testing metrics loading...")
    
    try:
        import evaluate
        
        # Test loading each metric
        metrics_to_test = ['bleu', 'rouge', 'accuracy', 'precision', 'recall', 'f1']
        
        for metric_name in metrics_to_test:
            try:
                metric = evaluate.load(metric_name)
                print(f"  ✓ {metric_name} loaded successfully")
            except Exception as e:
                print(f"  ❌ Failed to load {metric_name}: {e}")
                return False
        
        # Test perplexity separately as it requires module_type
        try:
            perplexity = evaluate.load('perplexity', module_type="metric")
            print("  ✓ perplexity loaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to load perplexity: {e}")
            return False
            
        print("  ✓ All metrics loaded successfully!")
        return True
        
    except ImportError:
        print("  ❌ evaluate library not available")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting evaluation system tests...\n")
    
    # Test metrics loading
    if not test_metrics_loading():
        print("\n❌ Metrics loading test failed!")
        sys.exit(1)
    
    print()
    
    # Test evaluation components
    try:
        test_evaluation_components()
    except Exception as e:
        print(f"\n❌ Evaluation components test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Evaluation system is ready to use.")
    print("\n💡 To run the full evaluation on trained models:")
    print("   python experiments/peft_vs_peft-ica/evaluate_models.py")

if __name__ == "__main__":
    main()
