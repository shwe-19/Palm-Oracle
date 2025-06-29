import json


keys = [
  "heart_short_shallow", "heart_short_medium", "heart_short_deep",
  "heart_medium_shallow", "heart_medium_medium", "heart_medium_deep",
  "heart_long_shallow", "heart_long_medium", "heart_long_deep",
  "head_short_shallow", "head_short_medium", "head_short_deep",
  "head_medium_shallow", "head_medium_medium", "head_medium_deep",
  "head_long_shallow", "head_long_medium", "head_long_deep",
  "life_short_shallow", "life_short_medium", "life_short_deep",
  "life_medium_shallow", "life_medium_medium", "life_medium_deep",
  "life_long_shallow", "life_long_medium", "life_long_deep",
  "fate_short_shallow", "fate_short_medium", "fate_short_deep",
  "fate_medium_shallow", "fate_medium_medium", "fate_medium_deep",
  "fate_long_shallow", "fate_long_medium", "fate_long_deep"
]


predictions = {}
for key in keys:
    predictions[key] = [f"{key} Prediction {i}" for i in range(1, 101)]


with open("model/line_predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("line_predictions.json has been generated with 100 predictions for each of the 36 keys.")
