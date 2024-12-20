import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Sample feedback phrases
feedback_positive = [
    "The Gulab Jamun was absolutely delightful!",
    "I loved the taste of Mysore Pak, it was heavenly!",
    "Rasgulla is my favorite sweet, so soft and sweet!",
    "The Sandesh was perfect for the festival!",
    "Mysore Pak has the best flavor, I can’t get enough!",
    "Kozhukattai is a must-have during Ganesh Chaturthi!",
    "Pitha was delicious, very satisfying!",
    "The Modak tasted amazing, just like homemade!",
    "Rasgulla was so fresh and delightful!",
    "I enjoyed the Kheer, it was rich and creamy!"
]

feedback_negative = [
    "I didn't like the Gulab Jamun, it was too sweet.",
    "Mysore Pak was dry and crumbly, not impressed.",
    "Rasgulla was not fresh, very disappointing.",
    "The Sandesh was too hard for my taste.",
    "Kozhukattai lacked flavor and texture.",
    "Pitha was bland, I expected better.",
    "The Modak tasted stale, not worth it.",
    "I found the Kheer too watery and flavorless.",
    "Gulab Jamun had an odd aftertaste.",
    "Mysore Pak was overly sweet and greasy."
]

# Creating a synthetic dataset
data_size = 5000  # Total number of data points
data = {
    "text": [],
    "label": []
}

# Generating synthetic feedback
for _ in range(data_size // 2):
    data["text"].append(np.random.choice(feedback_positive))
    data["label"].append(1)  # Positive sentiment

for _ in range(data_size // 2):
    data["text"].append(np.random.choice(feedback_negative))
    data["label"].append(0)  # Negative sentiment

# Creating a DataFrame
synthetic_data = pd.DataFrame(data)

# Save to CSV in a user-friendly directory
output_path = 'C:\\Users\\Rupesh Shinde\\Desktop\\CRM\\synthetic_feedback_data.csv'
synthetic_data.to_csv(output_path, index=False)

print(f"Synthetic data saved successfully to {output_path}")
