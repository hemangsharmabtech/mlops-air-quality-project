import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import time
import yaml
import json
from tqdm import tqdm

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """Load and prepare the air quality data"""
    print("📁 Loading dataset...")
    try:
        df = pd.read_csv("data/raw/air_index_quality.csv", low_memory=False)
        print(f"✅ Dataset loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure data/raw/air_index_quality.csv exists")
        raise

def preprocess_data(df):
    """Preprocess the data"""
    print("🔄 Preprocessing data...")
    
    # Select useful columns
    useful_cols = [
        'state_name', 'county_name', 'city_name', 'latitude', 'longitude',
        'parameter_name', 'sample_duration', 'pollutant_standard',
        'units_of_measure', 'arithmetic_mean', 'first_max_value',
        'ninety_eight_percentile', 'arithmetic_standard_dev',
        'observation_count', 'observation_percent', 'valid_day_count',
        'year', 'date_of_last_change'
    ]
    
    # Keep only useful columns and drop rows with missing critical values
    df = df[useful_cols].dropna(subset=['arithmetic_mean', 'latitude', 'longitude'])
    print(f"✅ Selected useful columns. Shape after cleaning: {df.shape}")
    
    # Convert date to numerical features
    df['date_of_last_change'] = pd.to_datetime(df['date_of_last_change'], errors='coerce')
    df['year'] = df['date_of_last_change'].dt.year
    df['month'] = df['date_of_last_change'].dt.month
    df.drop(columns=['date_of_last_change'], inplace=True)
    
    # Define target variable
    df['pollution_level'] = np.where(df['arithmetic_mean'] > 50, 1, 0)
    print(f"✅ Target variable created. Class distribution:\n{df['pollution_level'].value_counts()}")
    
    return df

def encode_categorical_features(df):
    """Encode categorical features using LabelEncoder"""
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    print("\n🔠 Encoding categorical columns:")
    for col in tqdm(cat_cols, desc="Encoding", unit="col"):
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def train_model():
    """Main training function"""
    
    # Load parameters
    params = load_params()
    train_params = params['train']
    
    print("🚀 Starting model training pipeline...")
    print(f"📊 Parameters: {train_params}")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    df = encode_categorical_features(df)
    
    # Prepare features and target
    X = df.drop(columns=['pollution_level', 'arithmetic_mean'])
    y = df['pollution_level']
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=train_params['test_size'], 
        stratify=y, 
        random_state=train_params['random_state']
    )
    
    print("✅ Train/Test Split Completed.")
    print(f"📈 Training samples: {X_train.shape[0]}")
    print(f"📊 Testing samples: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("🌲 Training Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=train_params['n_estimators'],
        max_depth=train_params['max_depth'],
        min_samples_split=train_params['min_samples_split'],
        min_samples_leaf=train_params['min_samples_leaf'],
        random_state=train_params['random_state'],
        n_jobs=-1
    )
    
    # Train with timing
    start = time.time()
    rf.fit(X_train, y_train)
    end = time.time()
    training_time = end - start
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    # Print results
    print(f"⏱ Training Time: {training_time:.2f} seconds")
    print("\n✅ Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = dict(zip(X.columns, rf.feature_importances_))
    top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    print("\n🔝 Top 10 Feature Importances:")
    for feature, importance in top_features.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/air_quality_model.pkl'
    joblib.dump(rf, model_path)
    print(f"\n💾 Model saved as {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'training_time_seconds': float(training_time),
        'confusion_matrix': cm.tolist(),
        'classification_report': cr,
        'feature_importance': feature_importance,
        'top_features': top_features,
        'dataset_info': {
            'original_shape': df.shape,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'feature_count': X.shape[1]
        },
        'model_parameters': train_params,
        'environment_info': {
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'sklearn_version': sklearn.__version__
        }
    }
    
    # Save metrics as JSON
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save simplified metrics for DVC
    with open('models/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Training_Time: {training_time:.2f} seconds\n")
        f.write(f"Train_Samples: {X_train.shape[0]}\n")
        f.write(f"Test_Samples: {X_test.shape[0]}\n")
        f.write(f"Feature_Count: {X.shape[1]}\n")
        f.write(f"NumPy_Version: {np.__version__}\n")
        f.write(f"Scikit_Learn_Version: {sklearn.__version__}\n")
    
    print("📊 Metrics saved to models/metrics.json and models/metrics.txt")
    
    return metrics

if __name__ == "__main__":
    try:
        import sklearn
        metrics = train_model()
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📈 Final Accuracy: {metrics['accuracy']:.4f}")
        print(f"🔧 Environment: NumPy {np.__version__}, scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        raise