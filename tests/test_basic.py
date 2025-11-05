import unittest
import os
import sys

class TestBasicSetup(unittest.TestCase):
    
    def test_model_file_exists(self):
        """Test that the model file exists"""
        print("Checking if model file exists...")
        model_exists = os.path.exists('models/air_quality_model.pkl')
        self.assertTrue(model_exists, "Model file should exist")
        print("✅ Model file exists")
    
    def test_data_file_exists(self):
        """Test that the data file exists"""
        print("Checking if data file exists...")
        data_exists = os.path.exists('data/raw/air_index_quality.csv')
        self.assertTrue(data_exists, "Data file should exist")
        print("✅ Data file exists")
    
    def test_requirements_exist(self):
        """Test that requirements.txt exists"""
        print("Checking if requirements.txt exists...")
        requirements_exists = os.path.exists('requirements.txt')
        self.assertTrue(requirements_exists, "requirements.txt should exist")
        print("✅ requirements.txt exists")
    
    def test_key_files_exist(self):
        """Test that key project files exist"""
        key_files = ['train.py', 'app.py', 'params.yaml', 'test.py']
        for file in key_files:
            with self.subTest(file=file):
                self.assertTrue(os.path.exists(file), f"{file} should exist")
        print("✅ All key project files exist")
        
    def test_dvc_files_exist(self):
        """Test that DVC tracking files exist"""
        dvc_files = [
            'data/raw/air_index_quality.csv.dvc',
            'models/air_quality_model.pkl.dvc'
        ]
        for file in dvc_files:
            with self.subTest(file=file):
                self.assertTrue(os.path.exists(file), f"{file} should exist")
        print("✅ All DVC tracking files exist")

if __name__ == '__main__':
    unittest.main(verbosity=2)
