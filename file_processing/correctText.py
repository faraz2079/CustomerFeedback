import json
from textblob import TextBlob

# Load the JSON file
input_file = "feedback_input_text_unique_variations_copy.json"
output_file = "corrected_feedback_data.json"

# Load the JSON data
with open(input_file, 'r') as file:
    data = json.load(file)

def correct_text_field(entry):
    text = entry["text"]
    corrected_text = str(TextBlob(text).correct())
    return {"text": corrected_text, "stars": entry["stars"]}


corrected_data = [correct_text_field(entry) for entry in data]

# Save the corrected data to a new file
with open(output_file, 'w') as file:
    json.dump(corrected_data, file, indent=4)

print(f"Corrected data saved to {output_file}")
