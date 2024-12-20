import pandas as pd
import numpy as np
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define regions, cultural events, and sweets
regions = ["North India", "South India", "East India", "West India"]
cultural_events = {
    "North India": ["Diwali", "Holi", "Lohri", "Baisakhi"],
    "South India": ["Pongal", "Onam", "Diwali", "Ugadi"],
    "East India": ["Durga Puja", "Diwali", "Bihu", "Poila Baisakh"],
    "West India": ["Ganesh Chaturthi", "Navratri", "Diwali", "Gudi Padwa"]
}

sweets = {
    "North India": ["Gulab Jamun", "Ladoo", "Kaju Katli", "Rasgulla"],
    "South India": ["Mysore Pak", "Payasam", "Kozhukattai", "Jangiri"],
    "East India": ["Sandesh", "Pitha", "Rasgulla", "Chomchom"],
    "West India": ["Modak", "Puran Poli", "Shrikhand", "Jalebi"]
}

# Function to generate random sentiment
def random_sentiment():
    return random.choice(["Positive", "Negative"])

# Create a synthetic dataset of 5000 records
data = []
for i in range(5000):
    customer_id = i + 1
    region = random.choice(regions)
    event = random.choice(cultural_events[region])
    sweet = random.choice(sweets[region])
    rating = random.randint(1, 5)
    sentiment = random_sentiment()
    
    data.append([customer_id, region, event, sweet, rating, sentiment])

# Create DataFrame
df = pd.DataFrame(data, columns=["CustomerID", "Region", "Cultural Event", "Sweet", "Rating", "Sentiment"])

# Save the dataset to a CSV file
df.to_csv(r"C:\Users\Rupesh Shinde\Desktop\CRM\synthetic.csv", index=False)

# Display the first few rows
print(df.head())
