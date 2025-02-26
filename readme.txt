RTL Logic Depth Predictor

A machine learning-based tool that predicts the combinational logic depth of signals in RTL modules without running a complete synthesis.

Overview

The RTL Logic Depth Predictor uses machine learning models to estimate the combinational logic depth of signals in RTL (Register Transfer Level) modules. This can be useful for early performance estimation and design optimization without the need for complete synthesis, which can be time-consuming for large designs.

Features

- Extracts relevant features from RTL code to predict logic depth
- Supports multiple machine learning models (Random Forest, XGBoost, Gradient Boosting)
- Command-line interface for training models and making predictions
- Supports both file-based and string-based RTL inputs
- Provides detailed prediction metrics and analysis

Installation

Prerequisites

- Python 3.6+
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib

Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
```

Usage

Training a Model

```bash
python rtl_logic_depth_predictor.py train --model_type ensemble --output model.joblib --samples 200
```

Parameters:
- `--model_type`: Type of model to train (options: 'rf', 'xgb', 'ensemble', default: 'ensemble')
- `--output`: Output file for the trained model (default: 'rtl_depth_model.joblib')
- `--samples`: Number of synthetic samples to generate for training (default: 100)

Making Predictions

```bash
python rtl_logic_depth_predictor.py predict --model model.joblib --rtl path/to/rtl_file.v --signal signal_name
```

Parameters:
- `--model`: Path to the trained model file
- `--rtl`: Path to the RTL file or RTL content
- `--signal`: Name of the signal to analyze
- `--verbose`: Flag to print detailed feature information

### Using as a Library

You can also use the predictor programmatically:

```python
from rtl_logic_depth_predictor import easy_predict

rtl_content = """
module example(
    input [7:0] a, b,
    output [7:0] out
);
    assign out = (a & b) | (a ^ b);
endmodule
"""

depth = easy_predict(rtl_content, 'out', 'model.joblib')
print(f"Logic depth for 'out' is {depth}")
```

How It Works

1. **Feature Extraction**: The predictor extracts relevant features from RTL code, including:
   - Signal occurrences and characteristics
   - Logic operations (AND, OR, XOR, etc.)
   - Expression complexity (nesting levels)
   - Fan-in and fan-out approximations
   - Module complexity indicators

2. **Model Training**: The extracted features are used to train a machine learning model using either synthetic data or real examples with known depths.

3. **Prediction**: For new RTL signals, the predictor extracts features and uses the trained model to predict the combinational logic depth.

Example

```bash
Train a model with 200 synthetic samples
python rtl_logic_depth_predictor.py train --samples 200 --output my_model.joblib

Predict logic depth for a signal in an RTL file
python rtl_logic_depth_predictor.py predict --model my_model.joblib --rtl designs/alu.v --signal result
```

Limitations

- Predictions are approximations and may not match exact synthesis results
- Complex constructs (generate blocks, complex always blocks) may have reduced accuracy
- The quality of predictions depends on the training data

Contributing

Contributions are welcome! Areas for improvement include:
- Better feature extraction for complex RTL constructs
- Support for additional RTL languages (SystemVerilog, VHDL)
- Integration with synthesis tools for validation
- Expanding the training dataset with real-world examples

