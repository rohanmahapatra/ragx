from datasets import load_dataset
import os

# Load and save the question dataset
data_dir = 'bioasq/'
dataset_name = 'bioasq'
question_dataset = load_dataset("jenhsia/ragged", dataset_name, split='train')

# Define the output path for the question JSON file
questions_json_path = os.path.join(data_dir, f"{dataset_name}_questions.json")

# Save the question dataset to a JSON file using the to_json method
question_dataset.to_json(questions_json_path, orient="records")

print(f"Question data saved to {questions_json_path}")