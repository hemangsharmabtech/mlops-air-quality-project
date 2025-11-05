import unittest
import sys
import os
import json

# Add the parent directory to Python path so we can import app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import app
import joblib
import numpy as np

class TestApp(unittest.TestCase):
    
    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_status_code(self):
        """Test that the home page returns 200 status code"""
        print("Testing home page...")
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Air Quality', response.data)
        print("✅ Home page test passed")
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        print("Testing health endpoint...")
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        # Parse JSON response
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        print("✅ Health endpoint test passed")
    
    def test_features_endpoint(self):
        """Test the features endpoint"""
        print("Testing features endpoint...")
        response = self.app.get('/features')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('features', data)
        self.assertIn('feature_count', data)
        self.assertIsInstance(data['features'], list)
        print("✅ Features endpoint test passed")
    
    def test_model_file_exists(self):
        """Test that the model file exists"""
        print("Checking if model file exists...")
        model_exists = os.path.exists('models/air_quality_model.pkl')
        self.assertTrue(model_exists, "Model file should exist")
        print("✅ Model file exists test passed")

class TestModel(unittest.TestCase):
    
    def test_model_can_load(self):
        """Test that the model can be loaded successfully"""
        print("Testing model loading...")
        try:
            model = joblib.load('models/air_quality_model.pkl')
            self.assertIsNotNone(model)
            print("✅ Model loading test passed")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")

class TestData(unittest.TestCase):
    
    def test_data_file_exists(self):
        """Test that the data file exists"""
        print("Checking if data file exists...")
        data_exists = os.path.exists('data/raw/air_index_quality.csv')
        self.assertTrue(data_exists, "Data file should exist")
        print("✅ Data file exists test passed")
    
    def test_data_can_load(self):
        """Test that the data can be loaded with pandas"""
        print("Testing data loading...")
        try:
            import pandas as pd
            df = pd.read_csv('data/raw/air_index_quality.csv')
            self.assertFalse(df.empty, "DataFrame should not be empty")
            print(f"✅ Data loading test passed. Shape: {df.shape}")
        except Exception as e:
            self.fail(f"Data loading failed: {e}")

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)