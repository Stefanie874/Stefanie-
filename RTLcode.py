import os
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
