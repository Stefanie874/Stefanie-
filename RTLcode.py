import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as XGBRegressor
import re
import time
import os
import joblib
import argparse
import sys

class RTLLogicDepthPredictor:
    """
    A class to predict combinational logic depth of signals in RTL modules
    without running a complete synthesis.
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.scaler = None
        
    def extract_features_from_rtl(self, rtl_module, target_signal):
        """
        Extract features from an RTL module for a specific target signal.
        
        Args:
            rtl_module (str): Path to the RTL module file or content as string
            target_signal (str): Name of the signal to analyze
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Read RTL content if it's a file path
        if os.path.exists(rtl_module):
            with open(rtl_module, 'r') as file:
                rtl_content = file.read()
        else:
            rtl_content = rtl_module
            
        features = {}
        
        # Extract basic features
        features['signal_name_length'] = len(target_signal)
        features['rtl_size'] = len(rtl_content)
        
        # Count signal occurrences
        features['signal_occurrences'] = rtl_content.count(target_signal)
        
        # Extract fan-in approximation (signals that feed into the target)
        signal_pattern = re.compile(r'assign\s+{}\s*=\s*([^;]+);'.format(re.escape(target_signal)))
        assign_match = signal_pattern.search(rtl_content)
        
        if assign_match:
            rhs = assign_match.group(1)
            # Count unique signals (words) in the right-hand side
            unique_signals = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', rhs))
            features['fan_in_count'] = len(unique_signals)
            
            # Count operators (complexity indicator)
            features['and_ops'] = rhs.count('&')
            features['or_ops'] = rhs.count('|')
            features['xor_ops'] = rhs.count('^')
            features['not_ops'] = rhs.count('~')
            features['add_ops'] = rhs.count('+')
            features['sub_ops'] = rhs.count('-')
            features['total_ops'] = (features['and_ops'] + features['or_ops'] + 
                                    features['xor_ops'] + features['not_ops'] +
                                    features['add_ops'] + features['sub_ops'])
                                    
            # Detect conditional expressions (often create MUXes)
            features['conditional_ops'] = rhs.count('?')
            
            # Detect parenthesis nesting level (expression complexity)
            max_nesting = 0
            current_nesting = 0
            for char in rhs:
                if char == '(':
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == ')':
                    current_nesting -= 1
            features['max_nesting'] = max_nesting
        else:
            # Handle case where target signal isn't a simple assign
            always_block_pattern = re.compile(r'always\s*@[^\n]+\n[^;]*{}\s*[<:]='.format(re.escape(target_signal)))
            always_match = always_block_pattern.search(rtl_content)
            
            if always_match:
                # It's in an always block, extract the surrounding block
                features['in_always_block'] = 1
                # Extract approximate complexity from always block
                surrounding_text = rtl_content[max(0, always_match.start()-100):min(len(rtl_content), always_match.end()+500)]
                features['fan_in_count'] = len(set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', surrounding_text))) - 1  # Subtract 1 for target
                features['and_ops'] = surrounding_text.count('&')
                features['or_ops'] = surrounding_text.count('|')
                features['xor_ops'] = surrounding_text.count('^')
                features['not_ops'] = surrounding_text.count('~')
                features['add_ops'] = surrounding_text.count('+')
                features['sub_ops'] = surrounding_text.count('-')
                features['total_ops'] = (features['and_ops'] + features['or_ops'] + 
                                        features['xor_ops'] + features['not_ops'] +
                                        features['add_ops'] + features['sub_ops'])
                features['conditional_ops'] = surrounding_text.count('?') + surrounding_text.count('if') + surrounding_text.count('case')
                features['max_nesting'] = 2  # Default assumption
            else:
                # Can't easily determine - use defaults
                features['in_always_block'] = 0
                features['fan_in_count'] = 1
                features['and_ops'] = 0
                features['or_ops'] = 0
                features['xor_ops'] = 0
                features['not_ops'] = 0
                features['add_ops'] = 0
                features['sub_ops'] = 0
                features['total_ops'] = 0
                features['conditional_ops'] = 0
                features['max_nesting'] = 0
        
        # Extract fan-out (signals that use this signal)
        fan_out_pattern = re.compile(r'[^a-zA-Z0-9_]{}[^a-zA-Z0-9_]'.format(re.escape(target_signal)))
        fan_out_matches = fan_out_pattern.findall(rtl_content)
        features['fan_out_approx'] = len(fan_out_matches)
        
        # Module complexity indicators
        features['module_flop_count'] = rtl_content.count('reg ') + rtl_content.count('always @(posedge')
        features['module_input_count'] = rtl_content.count('input ')
        features['module_output_count'] = rtl_content.count('output ')
        
        return features
    
    def prepare_training_data(self, datasets):
        """
        Prepare training data from a collection of datasets
        
        Args:
            datasets (list): List of dictionaries containing RTL, signal and actual depth
                            [{'rtl': 'module...', 'signal': 'sig_name', 'depth': 5}, ...]
                             
        Returns:
            tuple: X (features) and y (depths) for model training
        """
        X_data = []
        y_data = []
        
        for data in datasets:
            rtl = data['rtl']
            signal = data['signal']
            depth = data['depth']
            
            features = self.extract_features_from_rtl(rtl, signal)
            X_data.append(features)
            y_data.append(depth)
            
        # Convert to DataFrame for easier handling
        X = pd.DataFrame(X_data)
        y = np.array(y_data)
        
        return X, y
    
    def train(self, X, y, model_type='ensemble'):
        """
        Train the model to predict logic depth.
        
        Args:
            X (DataFrame): Features
            y (array): Target depths
            model_type (str): Type of model to use ('rf', 'xgb', 'ensemble')
            
        Returns:
            float: Model performance metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipelines with preprocessing
        self.scaler = StandardScaler()
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            # Simple parameters for faster training
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            }
        elif model_type == 'xgb':
            self.model = XGBRegressor(n_estimators=100, random_state=42)
            # Simple parameters for faster training
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5]
            }
        else:  # ensemble
            # Default to GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            # Simple parameters for faster training
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3]
            }
            
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='neg_mean_absolute_error'
        )
        
        # Time the training process
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Print results
        print(f"Model: {model_type}")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Return metrics as a dictionary
        return {
            'model_type': model_type,
            'best_params': grid_search.best_params_,
            'training_time': training_time,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'test_actual': y_test,
            'test_predicted': y_pred
        }
    
    def predict(self, rtl_module, target_signal):
        """
        Predict the combinational logic depth for a target signal
        
        Args:
            rtl_module (str): Path to RTL file or RTL content
            target_signal (str): Name of the signal to analyze
            
        Returns:
            float: Predicted logic depth
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Extract features
        start_time = time.time()
        features = self.extract_features_from_rtl(rtl_module, target_signal)
        feature_time = time.time() - start_time
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Scale features if needed
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
            X_scaled = self.model.named_steps['scaler'].transform(X)
            X_pred = pd.DataFrame(X_scaled, columns=X.columns)
        else:
            X_pred = X
        
        # Predict
        start_time = time.time()
        depth = self.model.predict(X_pred)[0]
        predict_time = time.time() - start_time
        
        # Round to nearest integer (as depth is usually an integer)
        rounded_depth = round(depth)
        
        print(f"Feature Extraction Time: {feature_time:.4f} seconds")
        print(f"Prediction Time: {predict_time:.4f} seconds")
        print(f"Total Time: {feature_time + predict_time:.4f} seconds")
        print(f"Predicted Logic Depth: {depth:.2f} -> {rounded_depth}")
        
        # Return both raw and rounded prediction
        return {
            'raw_depth': depth,
            'rounded_depth': rounded_depth,
            'feature_time': feature_time,
            'predict_time': predict_time,
            'features': features
        }
    
    def save_model(self, filename):
        """Save the trained model to file"""
        if self.model is None:
            raise ValueError("No trained model to save")
        joblib.dump(self.model, filename)
        
    def load_model(self, filename):
        """Load a trained model from file"""
        self.model = joblib.load(filename)

def generate_synthetic_data(num_samples=100):
    """Generate synthetic RTL data for training"""
    import random
    
    # Start with a few manually crafted examples for structure
    synthetic_data = [
        {
            'rtl': """
            module simple_adder(
                input [7:0] a, b,
                output [8:0] sum
            );
                assign sum = a + b;
            endmodule
            """,
            'signal': 'sum',
            'depth': 3
        },
        {
            'rtl': """
            module multiplexer(
                input sel,
                input [7:0] a, b,
                output [7:0] out
            );
                assign out = sel ? a : b;
            endmodule
            """,
            'signal': 'out',
            'depth': 2
        },
        {
            'rtl': """
            module complex_logic(
                input [7:0] a, b, c, d,
                output [7:0] result
            );
                assign result = ((a & b) | (c ^ d)) + (a - b);
            endmodule
            """,
            'signal': 'result',
            'depth': 5
        },
        {
            'rtl': """
            module sequential_logic(
                input clk, rst,
                input [7:0] data_in,
                output reg [7:0] data_out
            );
                always @(posedge clk or posedge rst) begin
                    if (rst)
                        data_out <= 8'b0;
                    else
                        data_out <= data_in + 8'b1;
                end
            endmodule
            """,
            'signal': 'data_out',
            'depth': 2
        },
        {
            'rtl': """
            module nested_logic(
                input [7:0] a, b, c, d, e, f,
                output [7:0] result
            );
                assign result = ((a & (b | c)) ^ (d & (e | f))) + ((a + b) * (c - d));
            endmodule
            """,
            'signal': 'result',
            'depth': 8
        }
    ]
    
    # Generate more synthetic examples with variations
    for i in range(num_samples - len(synthetic_data)):
        op_count = random.randint(1, 10)
        operations = []
        signals = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][:random.randint(2, 8)]
        
        # Generate random operations
        for _ in range(op_count):
            op_type = random.choice(['&', '|', '^', '+', '-', '*'])
            operands = random.sample(signals, 2)
            operations.append(f"({operands[0]} {op_type} {operands[1]})")
        
        # Create RTL with these operations
        result_expr = operations[0]
        for op in operations[1:]:
            combine_op = random.choice(['&', '|', '^', '+', '-'])
            result_expr = f"({result_expr} {combine_op} {op})"
        
        rtl = f"""
        module synthetic_{i}(
            input [{len(signals)*8-1}:0] {'_'.join(signals)},
            output [7:0] result
        );
            assign result = {result_expr};
        endmodule
        """
        
        # Approximate depth based on operation count and nesting
        depth = min(20, max(1, int(op_count * random.uniform(0.8, 1.2))))
        
        synthetic_data.append({
            'rtl': rtl,
            'signal': 'result',
            'depth': depth
        })
    
    return synthetic_data

def main():
    parser = argparse.ArgumentParser(description='RTL Logic Depth Predictor')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--model_type', choices=['rf', 'xgb', 'ensemble'], default='ensemble',
                             help='Type of model to train')
    train_parser.add_argument('--output', default='rtl_depth_model.joblib',
                             help='Output file for the trained model')
    train_parser.add_argument('--samples', type=int, default=100,
                             help='Number of synthetic samples to generate for training')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict logic depth for a signal')
    predict_parser.add_argument('--model', required=True,
                               help='Path to the trained model file')
    predict_parser.add_argument('--rtl', required=True,
                               help='Path to the RTL file or RTL content')
    predict_parser.add_argument('--signal', required=True,
                               help='Name of the signal to analyze')
    predict_parser.add_argument('--verbose', action='store_true',
                               help='Print detailed information')
    
    args = parser.parse_args()
    
    predictor = RTLLogicDepthPredictor()
    
    if args.command == 'train':
        print(f"Generating {args.samples} synthetic training samples...")
        synthetic_data = generate_synthetic_data(args.samples)
        
        print("Extracting features...")
        X, y = predictor.prepare_training_data(synthetic_data)
        
        print(f"Training {args.model_type} model...")
        metrics = predictor.train(X, y, model_type=args.model_type)
        
        print(f"Saving model to {args.output}...")
        predictor.save_model(args.output)
        print("Training complete!")
        
    elif args.command == 'predict':
        print(f"Loading model from {args.model}...")
        predictor.load_model(args.model)
        
        print(f"Reading RTL from {args.rtl}...")
        if not os.path.exists(args.rtl):
            print("Error: RTL file not found")
            return
            
        print(f"Predicting logic depth for signal '{args.signal}'...")
        result = predictor.predict(args.rtl, args.signal)
        
        print("\n--- PREDICTION RESULTS ---")
        print(f"Signal: {args.signal}")
        print(f"Predicted Logic Depth: {result['rounded_depth']}")
        print(f"Confidence (Raw Value): {result['raw_depth']:.2f}")
        print(f"Prediction Time: {result['feature_time'] + result['predict_time']:.4f} seconds")
        
        if args.verbose:
            print("\nExtracted Features:")
            for feat, value in result['features'].items():
                print(f"  {feat}: {value}")
    
    else:
        parser.print_help()

def easy_predict(rtl_content, signal_name, model_path=None):
    """
    Simplified function to predict logic depth from RTL content
    
    Args:
        rtl_content (str): The RTL code as a string
        signal_name (str): The signal to analyze
        model_path (str): Path to a pre-trained model, or None to train a new one
        
    Returns:
        int: Predicted logic depth
    """
    predictor = RTLLogicDepthPredictor()
    
    if model_path and os.path.exists(model_path):
        # Load existing model
        predictor.load_model(model_path)
    else:
        # Train a new model
        print("Training a new model...")
        synthetic_data = generate_synthetic_data(100)
        X, y = predictor.prepare_training_data(synthetic_data)
        predictor.train(X, y)
        
        # Save the model for future use
        if model_path:
            predictor.save_model(model_path)
    
    # Make prediction
    result = predictor.predict(rtl_content, signal_name)
    return result['rounded_depth']

if __name__ == "__main__":
    # If run as a script, use the command-line interface
    if len(sys.argv) > 1:
        main()
    else:
        # Simple example for direct import usage
        rtl_example = """
        module example(
            input [7:0] a, b,
            output [7:0] out
        );
            assign out = (a & b) | (a ^ b);
        endmodule
        """
        
        depth = easy_predict(rtl_example, 'out')
        print(f"Example prediction: Logic depth for 'out' is {depth}")import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import joblib
import time

class RTLDepthPredictor:
    def __init__(self, model_type='xgboost'):
        """
        Initialize the RTL Depth Predictor with the specified model type.
        
        Parameters:
        -----------
        model_type : str
            The type of model to use ('random_forest', 'gradient_boosting', or 'xgboost')
        """
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'xgboost':
            self.model = XGBRegressor(random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose from 'random_forest', 'gradient_boosting', or 'xgboost'.")
        
        self.feature_names = None
        self.prediction_time = 0
    
    def extract_features_from_rtl(self, rtl_file, signal_name):
        """
        Extract features from an RTL file for a specific signal.
        
        Parameters:
        -----------
        rtl_file : str
            Path to the RTL file
        signal_name : str
            Name of the signal to analyze
            
        Returns:
        --------
        dict
            Dictionary of extracted features
        """
        with open(rtl_file, 'r') as f:
            rtl_content = f.read()
        
        # Create a graph representation of the circuit
        G = self._build_circuit_graph(rtl_content)
        
        # Extract features
        features = {}
        
        # 1. Basic signal properties
        features['signal_name_length'] = len(signal_name)
        features['is_registered'] = 1 if re.search(r'reg\s+' + signal_name, rtl_content) else 0
        
        # 2. Fan-in analysis
        pred = list(G.predecessors(signal_name)) if signal_name in G else []
        features['fan_in'] = len(pred)
        features['max_fan_in_depth'] = self._get_max_depth(G, signal_name)
        
        # 3. Fan-out analysis (adding as requested)
        succ = list(G.successors(signal_name)) if signal_name in G else []
        features['fan_out'] = len(succ)
        
        # 4. Operation complexity
        operations = ['&', '|', '^', '~', '+', '-', '*', '/', '<<', '>>', '?']
        for op in operations:
            pattern = r'\b' + re.escape(signal_name) + r'\s*' + re.escape(op)
            features[f'op_{op}_count'] = len(re.findall(pattern, rtl_content))
        
        # 5. Conditional complexity
        features['if_count'] = len(re.findall(r'if\s*\(.+?' + re.escape(signal_name) + r'.+?\)', rtl_content))
        features['case_count'] = len(re.findall(r'case\s*\(.+?' + re.escape(signal_name) + r'.+?\)', rtl_content))
        
        # 6. Module complexity
        features['module_loc'] = len(rtl_content.split('\n'))
        features['always_blocks'] = len(re.findall(r'always\s*@', rtl_content))
        
        # 7. Signal width (if available)
        width_match = re.search(r'(\[\d+:\d+\])\s+' + re.escape(signal_name), rtl_content)
        if width_match:
            width_str = width_match.group(1)
            msb, lsb = map(int, re.findall(r'\d+', width_str))
            features['signal_width'] = abs(msb - lsb) + 1
        else:
            features['signal_width'] = 1  # Default to 1-bit
        
        # 8. Circuit topology
        if signal_name in G:
            features['centrality'] = nx.degree_centrality(G)[signal_name]
            features['closeness'] = nx.closeness_centrality(G)[signal_name]
            try:
                features['betweenness'] = nx.betweenness_centrality(G)[signal_name]
            except:
                features['betweenness'] = 0
        else:
            features['centrality'] = 0
            features['closeness'] = 0
            features['betweenness'] = 0
            
        return features
    
    def _build_circuit_graph(self, rtl_content):
        """
        Build a graph representation of the circuit from RTL content.
        
        Parameters:
        -----------
        rtl_content : str
            RTL content as a string
            
        Returns:
        --------
        nx.DiGraph
            Directed graph representing the circuit
        """
        G = nx.DiGraph()
        
        # Extract all signal declarations
        signal_pattern = r'(input|output|wire|reg)\s+(?:\[\d+:\d+\])?\s*([a-zA-Z0-9_]+)'
        signals = re.findall(signal_pattern, rtl_content)
        
        for _, signal in signals:
            G.add_node(signal.strip())
        
        # Extract assignments
        assign_pattern = r'assign\s+([a-zA-Z0-9_]+)\s*=\s*(.+?);'
        assignments = re.findall(assign_pattern, rtl_content)
        
        for lhs, rhs in assignments:
            lhs = lhs.strip()
            # Find all signals in the right-hand side
            rhs_signals = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', rhs)
            for rhs_signal in rhs_signals:
                if rhs_signal in G:
                    G.add_edge(rhs_signal, lhs)
        
        # Extract always block assignments
        always_blocks = re.findall(r'always\s*@[^;]+\s*begin(.*?)end', rtl_content, re.DOTALL)
        for block in always_blocks:
            assignments = re.findall(r'([a-zA-Z0-9_]+)\s*<=\s*(.+?);', block)
            for lhs, rhs in assignments:
                lhs = lhs.strip()
                rhs_signals = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', rhs)
                for rhs_signal in rhs_signals:
                    if rhs_signal in G:
                        G.add_edge(rhs_signal, lhs)
        
        return G
    
    def _get_max_depth(self, G, node):
        """
        Calculate the maximum depth of a node in the graph.
        
        Parameters:
        -----------
        G : nx.DiGraph
            Graph representing the circuit
        node : str
            Node to find the depth for
            
        Returns:
        --------
        int
            Maximum depth of the node
        """
        if node not in G:
            return 0
            
        pred = list(G.predecessors(node))
        if not pred:
            return 0
            
        return 1 + max(self._get_max_depth(G, p) for p in pred)
    
    def create_dataset(self, rtl_files, signal_names, true_depths):
        """
        Create a dataset from multiple RTL files and their corresponding signals.
        
        Parameters:
        -----------
        rtl_files : list
            List of paths to RTL files
        signal_names : list
            List of signal names to analyze for each file
        true_depths : list
            List of true combinational depths for each signal
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with features and true depths
        """
        data = []
        
        for i in tqdm(range(len(rtl_files)), desc="Processing RTL files"):
            features = self.extract_features_from_rtl(rtl_files[i], signal_names[i])
            features['true_depth'] = true_depths[i]
            data.append(features)
        
        return pd.DataFrame(data)
    
    def train(self, X, y):
        """
        Train the model on the given data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features
        y : pandas.Series
            Target values (true combinational depths)
        """
        self.feature_names = X.columns.tolist()
        
        # Parameter grid for hyperparameter tuning
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:  # xgboost
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X, y)
        
        # Set the model to the best estimator
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    
    def predict(self, X):
        """
        Predict combinational depths using the trained model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features
            
        Returns:
        --------
        numpy.ndarray
            Predicted combinational depths
        """
        start_time = time.time()
        predictions = self.model.predict(X)
        end_time = time.time()
        
        self.prediction_time = end_time - start_time
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            True depths for test data
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        results = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'Prediction Time (s)': self.prediction_time
        }
        
        # Calculate percentage of predictions within +/- 1 of true value
        within_one = np.mean(np.abs(y_test - y_pred) <= 1)
        results['Within +/- 1'] = within_one
        
        return results
    
    def feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importance values
        """
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
            
        return pd.DataFrame({'Feature': self.feature_names, 'Importance': importance}) \
                .sort_values('Importance', ascending=False) \
                .reset_index(drop=True)
    
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filename : str
            Path to save the model
        """
        joblib.dump(self.model, filename)
    
    def load_model(self, filename):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filename : str
            Path to the saved model
        """
        self.model = joblib.load(filename)

# Example usage
if __name__ == "__main__":
    # In a real scenario, you would have actual RTL files and synthesis data
    # For demonstration, we'll create a simulated dataset
    
    # 1. Create simulated data (in a real scenario, this would be actual RTL files and synthesis data)
    np.random.seed(42)
    num_samples = 100
    
    # Simulate feature values
    fan_in_values = np.random.randint(1, 20, num_samples)
    fan_out_values = np.random.randint(1, 15, num_samples)  # Added fan-out
    signal_width_values = np.random.choice([1, 8, 16, 32, 64], num_samples)
    op_count_values = np.random.randint(0, 10, num_samples)
    if_count_values = np.random.randint(0, 5, num_samples)
    
    # Create a DataFrame with simulated features
    X = pd.DataFrame({
        'fan_in': fan_in_values,
        'fan_out': fan_out_values,  # Added fan-out feature
        'signal_width': signal_width_values,
        'op_&_count': op_count_values,
        'if_count': if_count_values,
        'is_registered': np.random.choice([0, 1], num_samples),
        'module_loc': np.random.randint(100, 1000, num_samples),
        'always_blocks': np.random.randint(1, 10, num_samples)
    })
    
    # Simulate target values based on features with some noise
    # Formula: depth ≈ 0.8 * fan_in + 0.3 * op_count + 0.2 * if_count + noise
    y = (0.8 * X['fan_in'] + 0.2 * X['fan_out'] + 0.3 * X['op_&_count'] + 
         0.2 * X['if_count'] + 0.1 * X['always_blocks'] + 
         np.random.normal(0, 1, num_samples)).astype(int)
    y = np.maximum(y, 1)  # Depth should be at least 1
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Initialize the RTL Depth Predictor
    predictor = RTLDepthPredictor(model_type='xgboost')
    
    # 3. Train the model
    print("Training the model...")
    predictor.train(X_train, y_train)
    
    # 4. Evaluate the model
    print("\nEvaluating the model...")
    results = predictor.evaluate(X_test, y_test)
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    
    # 5. Analyze feature importance
    importance_df = predictor.feature_importance()
    print("\nFeature Importance:")
    print(importance_df.head(10))
    
    # 6. Make predictions on new data
    print("\nPredicting on new data...")
    new_data = pd.DataFrame({
        'fan_in': [10],
        'fan_out': [8],  # Added fan-out
        'signal_width': [32],
        'op_&_count': [5],
        'if_count': [2],
        'is_registered': [1],
        'module_loc': [500],
        'always_blocks': [3]
    })
    
    predicted_depth = predictor.predict(new_data)[0]
    print(f"Predicted combinational depth: {predicted_depth}")
    
    # 7. Save the model
    predictor.save_model("rtl_depth_predictor.joblib")
    print("\nModel saved to 'rtl_depth_predictor.joblib'")
    
    # 8. Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot predictions vs. actual values
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictor.predict(X_test))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.title('Prediction vs. Actual')
    
    # Plot feature importance
    plt.subplot(1, 2, 2)
    importance_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('rtl_depth_prediction_results.png')
    plt.close()
    
    print("Results visualization saved to 'rtl_depth_prediction_results.png'")

# Function to run prediction on a new RTL file and signal
def predict_depth_for_rtl(rtl_file_path, signal_name, model_path="rtl_depth_predictor.joblib"):
    """
    Predict the combinational depth for a given signal in an RTL file.
    
    Parameters:
    -----------
    rtl_file_path : str
        Path to the RTL file
    signal_name : str
        Name of the signal to analyze
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    int
        Predicted combinational depth
    float
        Prediction time in seconds
    """
    predictor = RTLDepthPredictor()
    predictor.load_model(model_path)
    
    features = predictor.extract_features_from_rtl(rtl_file_path, signal_name)
    
    # Convert features to DataFrame (excluding true_depth if present)
    if 'true_depth' in features:
        del features['true_depth']
    
    feature_df = pd.DataFrame([features])
    
    start_time = time.time()
    predicted_depth = predictor.predict(feature_df)[0]
    end_time = time.time()
    
    return int(round(predicted_depth)), end_time - start_time
