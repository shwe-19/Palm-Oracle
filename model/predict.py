import cv2
import numpy as np
import joblib
import random
import os
from model.utils import get_prediction_statement

# Line identifiers
LINE_CODES = {
    'heart': 0,
    'head': 1,
    'life': 2,
    'fate': 3
}

# HSV color ranges for line detection
COLOR_RANGES = {
    'heart': ([0, 50, 50], [10, 255, 255]),              # Red
    'head':  ([30, 20, 20], [95, 255, 255]),             # Green
    'life':  ([125, 40, 40], [160, 255, 255]),           # Purple
    'fate':  ([90, 40, 40], [135, 255, 255])             # Blue
}

# Default feature fallback
DEFAULT_FEATURES = {
    'heart': {'arc_length': 500, 'area': 1000, 'length': 'long', 'depth': 'deep'},
    'head':  {'arc_length': 160, 'area': 2800, 'length': 'long', 'depth': 'moderate'},
    'life':  {'arc_length': 400, 'area': 2000, 'length': 'long', 'depth': 'deep'},
    'fate':  {'arc_length': 200, 'area': 500,  'length': 'short', 'depth': 'shallow'}
}

# BGR colors for drawing lines
LINE_COLORS = {
    'heart': (0, 0, 255),    # Red
    'head': (0, 255, 0),     # Green
    'life': (255, 0, 255),   # Purple
    'fate': (255, 0, 0)      # Blue
}

def analyze_palm_image(image_path, gender=None, hand=None, output_path=None):
    try:
        model = joblib.load("model/saved_model.pkl")
    except Exception as e:
        return {"error": f"Model could not be loaded: {e}"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Image could not be processed."}

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    result = {}

    for line in COLOR_RANGES:
        lower, upper = COLOR_RANGES[line]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            arc_length = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)

            feature = [[arc_length, area, LINE_CODES[line]]]
            prediction = model.predict(feature)[0]

            # Draw the contour on the image
            cv2.drawContours(image, [cnt], -1, LINE_COLORS[line], 2)

            # Decode the model output
            if "_" in prediction:
                length_cat, depth_cat = prediction.split("_")
            else:
                length_cat, depth_cat = "unknown", "unknown"

            # Get life predictions
            predictions_all = get_prediction_statement(line, length_cat, depth_cat)
            if not predictions_all:
                predictions_all = [f"Default prediction for {line} line with {length_cat} length and {depth_cat} depth."]

            # Select 6 random predictions
            while len(predictions_all) < 6:
                predictions_all *= 2
            predictions = random.sample(predictions_all[:6], 6)

            result[line] = {
                'length': f"{length_cat} ({int(arc_length)} px)",
                'depth': f"{depth_cat} ({int(area)} px²)",
                'predictions': predictions
            }

        else:
            # Fallback to default values
            default = DEFAULT_FEATURES[line]
            arc_length = default['arc_length']
            area = default['area']
            length_cat = default['length']
            depth_cat = default['depth']

            predictions_all = get_prediction_statement(line, length_cat, depth_cat)
            if not predictions_all:
                predictions_all = [f"Default prediction for {line} line with {length_cat} length and {depth_cat} depth."]

            while len(predictions_all) < 6:
                predictions_all *= 2
            predictions = random.sample(predictions_all[:6], 6)

            result[line] = {
                'length': f"{length_cat} ({arc_length} px)",
                'depth': f"{depth_cat} ({area} px²)",
                'predictions': predictions
            }

    # Save the image with drawn lines
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        result['marked_image'] = output_path

    return result

# For testing
if __name__ == "__main__":
    image_path = "C:/Users/SHWETA/Desktop/Palmproject/static/images/uploads/sample.jpg"
    output_path = "C:/Users/SHWETA/Desktop/Palmproject/static/images/marked/marked_sample.jpg"

    analysis = analyze_palm_image(image_path, output_path=output_path)

    print(f"✅ Marked image saved to: {analysis.get('marked_image')}\n")
    for line, data in analysis.items():
        if line == 'marked_image':
            continue
        print(f"{line.upper()} LINE:")
        print(f"Length: {data['length']}")
        print(f"Depth: {data['depth']}")
        print("Life Predictions:")
        for statement in data['predictions']:
            print("-", statement)
        print("\n") 