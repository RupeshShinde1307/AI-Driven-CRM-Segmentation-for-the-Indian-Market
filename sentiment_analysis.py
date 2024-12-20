import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset

# Define a custom dataset class
class FeedbackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load dataset
file_path = r'C:\Users\Rupesh Shinde\Desktop\CRM\synthetic.csv'  # Replace with your actual path
data = pd.read_csv(file_path)

# Display first few rows of the dataset for reference
print(data.head())

# Split the dataset into training and testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['Feedback'].tolist(), data['Sentiment'].map({"Positive": 1, "Negative": 0}).tolist(), test_size=0.2, random_state=42
)

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create datasets
train_dataset = FeedbackDataset(train_encodings, train_labels)
test_dataset = FeedbackDataset(test_encodings, test_labels)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch"      # evaluate every epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
def evaluate_model(model, eval_dataset):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in DataLoader(eval_dataset, batch_size=8):
            inputs = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(inputs)
            logits = outputs.logits
            
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

# Call evaluation function
evaluate_model(model, test_dataset)

# User input for prediction
feedback = input("Enter feedback for sentiment prediction: ")
inputs = tokenizer(feedback, return_tensors='pt', truncation=True, padding=True, max_length=128).to(model.device)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()

# Map label back to sentiment
sentiment = "Positive" if predicted_label == 1 else "Negative"
print(f"Predicted Sentiment: {sentiment}")
