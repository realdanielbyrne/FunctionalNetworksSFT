#!/usr/bin/env python3
"""
Comprehensive test suite for ICA masking functionality.

This script runs all ICA-related tests and provides a summary report.
"""

import sys
import unittest
import logging
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_ica_tests():
    """Run all ICA-related tests and return results."""
    logger.info("=" * 60)
    logger.info("ICA MASKING FUNCTIONALITY TEST SUITE")
    logger.info("=" * 60)

    # Import test modules
    try:
        from tests.test_ica_masking import (
            TestICAMasking,
            TestICAMaskComputation,
            TestICAMaskingIntegration,
        )
        from tests.test_ica_cli_integration import (
            TestICACommandLineArguments,
            TestICATrainingIntegration,
            TestICAErrorHandling,
        )
        from tests.test_ica_mask_application import (
            TestMaskApplication,
            TestMaskHookBehavior,
        )

        logger.info("OK Successfully imported all test modules")
    except ImportError as e:
        logger.error(f"X Failed to import test modules: {e}")
        return False

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        # Core ICA functionality tests
        TestICAMasking,
        TestICAMaskComputation,
        TestICAMaskingIntegration,
        # CLI integration tests
        TestICACommandLineArguments,
        TestICATrainingIntegration,
        TestICAErrorHandling,
        # Mask application tests
        TestMaskApplication,
        TestMaskHookBehavior,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, "skipped") else 0
    passed = total_tests - failures - errors - skipped

    logger.info(f"Total tests run: {total_tests}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failures}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Skipped: {skipped}")

    if failures > 0:
        logger.info("\nFAILURES:")
        for test, traceback in result.failures:
            logger.info(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if errors > 0:
        logger.info("\nERRORS:")
        for test, traceback in result.errors:
            logger.info(f"- {test}: {traceback.split('Exception:')[-1].strip()}")

    success = failures == 0 and errors == 0

    if success:
        logger.info("\nAll ICA tests passed! The functionality is working correctly.")
    else:
        logger.error("\nSome ICA tests failed. Please check the implementation.")

    return success


def test_dependencies():
    """Test that all required dependencies for ICA functionality are available."""
    logger.info("Testing ICA dependencies...")

    dependencies = [
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("torch", "PyTorch"),
        ("json", "json (built-in)"),
        ("collections", "collections (built-in)"),
        ("itertools", "itertools (built-in)"),
    ]

    missing_deps = []

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            logger.info(f"OK {display_name}")
        except ImportError:
            logger.error(f"X {display_name}")
            missing_deps.append(display_name)

    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies before running ICA tests.")
        return False

    # Test specific sklearn components
    try:
        from sklearn.decomposition import FastICA

        logger.info("OK FastICA from scikit-learn")
    except ImportError:
        logger.error("X FastICA not available in scikit-learn")
        return False

    return True


def test_basic_functionality():
    """Test basic ICA functionality without full test suite."""
    logger.info("Testing basic ICA functionality...")

    try:
        # Test imports
        from src.functionalnetworkssft.fnsft_trainer import (
            apply_ica_masks,
            compute_ica_masks_for_model,
        )

        logger.info("OK ICA functions imported successfully")

        # Test basic mask creation
        import torch
        import torch.nn as nn

        # Create simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {})()
                self.config.hidden_size = 768
                self.model = type("Model", (), {})()
                self.model.layers = nn.ModuleList(
                    [
                        nn.Linear(
                            768, 3072
                        ),  # This won't match the pattern, but that's OK for basic test
                    ]
                )
                self.embedding = nn.Embedding(1000, 768)

            def get_input_embeddings(self):
                return self.embedding

        model = SimpleModel()
        mask_dict = {"0": [100, 200, 300]}

        # Test mask application (should not crash)
        handles = apply_ica_masks(model, mask_dict, mask_mode="key")
        logger.info("OK Basic mask application works")

        # Clean up
        for handle in handles:
            handle.remove()

        return True

    except Exception as e:
        logger.error(f"X Basic functionality test failed: {e}")
        return False


def main():
    """Main test runner function."""
    logger.info("Starting ICA masking test suite...")

    # Test dependencies first
    if not test_dependencies():
        logger.error("Dependency test failed. Cannot proceed with ICA tests.")
        return 1

    # Test basic functionality
    if not test_basic_functionality():
        logger.error(
            "Basic functionality test failed. Cannot proceed with full test suite."
        )
        return 1

    # Run full test suite
    try:
        success = run_ica_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
