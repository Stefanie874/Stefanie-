# RTL Logic Depth Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-based tool that predicts the combinational logic depth of signals in RTL (Register Transfer Level) Verilog code without requiring full synthesis. This tool helps hardware designers identify potential timing bottlenecks early in the design process.

## Overview

Hardware designers often need to estimate critical path delays in their designs before running a full synthesis flow. The RTL Logic Depth Predictor analyzes Verilog code and predicts the combinational logic depth of signals based on various code features, providing insights into potential timing issues.

Key features:
- Extract relevant features from Verilog RTL code
- Predict combinational logic depth using either ML models or heuristics
- Identify potential timing bottlenecks
- Analyze fan-in and fan-out relationships
- Train custom models on your specific design patterns

## Installation

### Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rtl-logic-depth-predictor.git
cd rtl-logic-depth-predictor
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## Usage

### Basic Usage

Analyze a Verilog file to predict logic depth for all signals:

```bash
python rtl_logic_depth_predictor.py --rtl path/to/your/file.v
```

Analyze a specific signal in a Verilog file:

```bash
python rtl_logic_depth_predictor.py --rtl path/to/your/file.v --signal signal_name
```

### Training a Model

Train a new model on synthetic data:

```bash
python rtl_logic_depth_predictor.py --train --model custom_model.joblib
```

Use a trained model for prediction:

```bash
python rtl_logic_depth_predictor.py --rtl path/to/your/file.v --model custom_model.joblib
```

## How It Works

The tool extracts the following features from RTL code:

1. **Operation Count**: Number of operations in the signal definition
2. **Fan-in**: Number of signals feeding into the target signal
3. **Fan-out**: Number of signals dependent on the target signal
4. **Nesting Depth**: Maximum depth of nested operations
5. **Conditional Statements**: Count of conditional constructs (if, case, ternary)
6. **Arithmetic Operations**: Count of arithmetic operators (+, -, *, /, %)
7. **Logical Operations**: Count of logical operators (&, |, ^, etc.)
8. **Signal Name Complexity**: Length of signal name
9. **Module Complexity**: Size and line count of the module
10. **Assignment Complexity**: Complexity of assignments
11. **Mux Complexity**: Count of ternary operators (? :)
12. **Bit Width**: Estimated width of the signal
13. **Function Calls**: Count of function calls in the signal definition

These features are then used by a machine learning model (RandomForest or GradientBoosting) to predict the combinational logic depth. If no trained model is available, a heuristic estimation is used.

## Example Output

```
Analysis results:
------------------------------------------------------------
Signal          Depth   Fan-In  Fan-Out
------------------------------------------------------------
data_out        8       3       0
temp1           4       2       1
temp2           4       2       1
intermediate    3       1       1
valid_out       2       1       0

Potential timing bottlenecks (high depth and fan-out):
data_out: Depth=8, Fan-out=0
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool uses machine learning techniques from scikit-learn
- Thanks to the hardware design community for feedback and testing
