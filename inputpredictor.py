#!/usr/bin/env python3
"""
RTL Logic Depth Predictor - Command Line Interface
Usage:
  python rtl_predictor.py <rtl_file> <signal_name>
"""

import sys
import os
from rtl_logic_depth_predictor import RTLLogicDepthPredictor, generate_synthetic_data

def main():
    # Check if we have the correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python rtl_predictor.py <rtl_file> <signal_name>")
        sys.exit(1)
    
    rtl_file = sys.argv[1]
    signal_name = sys.argv[2]
    
    # Check if the RTL file exists
    if not os.path.exists(rtl_file):
        print(f"Error: RTL file '{rtl_file}' not found")
        sys.exit(1)
    
    # Read the RTL file
    with open(rtl_file, 'r') as f:
        rtl_content = f.read()
    
    # Print information about the file
    print(f"Processing RTL file: {rtl_file}")
    print(f"File size: {len(rtl_content)} bytes")
    print(f"Target signal: {signal_name}")
    
    # Create the predictor
    predictor = RTLLogicDepthPredictor()
    
    # Check if we have a pre-trained model
    model_path = "rtl_depth_model.joblib"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        predictor.load_model(model_path)
    else:
        print("No pre-trained model found. Training a new model...")
        print("This may take a moment...")
        
        # Generate synthetic data for training
        print("Generating synthetic training data...")
        synthetic_data = generate_synthetic_data(100)
        
        # Extract features and train the model
        print("Extracting features and training model...")
        X, y = predictor.prepare_training_data(synthetic_data)
        predictor.train(X, y, model_type='ensemble')
        
        # Save the model for future use
        print(f"Saving model to {model_path}")
        predictor.save_model(model_path)
    
    # Make the prediction
    print("\nAnalyzing RTL and predicting logic depth...")
    result = predictor.predict(rtl_content, signal_name)
    
    # Display results
    print("\n========== PREDICTION RESULTS ==========")
    print(f"Signal: {signal_name}")
    print(f"Predicted Logic Depth: {result['rounded_depth']}")
    print("=========================================")
    
    # Optional: export the results to a file
    with open(f"{signal_name}_depth_prediction.txt", "w") as f:
        f.write(f"RTL file: {rtl_file}\n")
        f.write(f"Signal: {signal_name}\n")
        f.write(f"Predicted Logic Depth: {result['rounded_depth']}\n")
        f.write(f"Raw Prediction Value: {result['raw_depth']:.2f}\n")
        f.write(f"Feature Extraction Time: {result['feature_time']:.4f} seconds\n")
        f.write(f"Prediction Time: {result['predict_time']:.4f} seconds\n")
    
    print(f"Results saved to {signal_name}_depth_prediction.txt")

if __name__ == "__main__":
    main()
