import os
import json

# Get absolute path to line_predictions.json inside the model folder
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'line_predictions.json')

# Load predictions from the JSON file
try:
    with open(file_path, encoding='utf-8') as f:
        predictions = json.load(f)
except Exception as e:
    print(f"Error loading prediction file: {e}")
    predictions = {}

def get_prediction_statement(line_type, length, depth):
    """
    Fetches prediction statements for a given palm line type and predicted length & depth.
    Example key: 'head_long_medium'
    """
    key = f"{line_type}_{length}_{depth}"
    return predictions.get(key, [])
