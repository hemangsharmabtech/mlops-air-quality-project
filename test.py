#!/usr/bin/env python3
"""
Main test runner for Air Quality MLOps Project
Runs all unit tests and returns exit code for CI/CD
"""

import unittest
import sys
import os

def run_tests():
    """
    Discover and run all tests in the tests directory
    """
    print("=" * 60)
    print("🚀 Running Air Quality MLOps Test Suite")
    print("=" * 60)
    
    # Add current directory to Python path
    sys.path.append(os.path.dirname(__file__))
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False

if __name__ == '__main__':
    success = run_tests()
    
    # Exit with proper code for CI/CD (0 = success, 1 = failure)
    sys.exit(0 if success else 1)