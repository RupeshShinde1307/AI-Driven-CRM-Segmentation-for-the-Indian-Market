# Attempting to generate the synthetic data file again after previous issues
import pandas as pd
import random
import numpy as np

# Number of data points
num_points_5000 = 5000

# Defining regions and preferences
regions = ['North India', 'South India', 'East India', 'West India']
sweets = {
    'North India': ['Gulab Jamun', 'Rasgulla', 'Ladoo', 'Barfi'],
    'South India': ['Mysore Pak', 'Kozhukattai', 'Payasam', 'Boondi Laddu'],
    'East India': ['Sandesh', 'Pitha', 'Rasgulla', 'Chhena Poda'],
    'West India': ['Mohanthal', 'Puran Poli', 'Shrikhand', 'Ladoo']
}
festivals = {
    'North India': ['Diwali', 'Holi', 'Baisakhi', 'Karva Chauth'],
    'South India': ['Onam', 'Pongal', 'Ugadi', 'Diwali'],
    'East India': ['Durga Puja', 'Makar Sankranti', 'Rath Yatra', 'Poila Baisakh'],
    'West India': ['Ganesh Chaturthi', 'Navratri', 'Gudi Padwa', 'Diwali']
}
languages = {
    'North India': ['Hindi', 'Punjabi', 'Urdu'],
    'South India': ['Tamil', 'Telugu', 'Malayalam', 'Kannada'],
    'East India': ['Bengali', 'Odia', 'Assamese'],
    'West India': ['Marathi', 'Gujarati', 'Konkani']
}
socio_economic_classes = ['Lower', 'Middle', 'Upper']

# Generate synthetic data
data_5000 = []
for _ in range(num_points_5000):
    region = random.choice(regions)
    sweet_preference = random.choice(sweets[region])
    festival = random.choice(festivals[region])
    language = random.choice(languages[region])
    socio_economic_class = random.choice(socio_economic_classes)
    purchase_amount = round(np.random.uniform(100, 1000), 2)  # Random amount between 100 and 1000

    data_5000.append([region, sweet_preference, festival, language, socio_economic_class, purchase_amount])

# Create DataFrame
columns = ['Region', 'Sweet_Preference', 'Festival', 'Language', 'Socio_Economic_Class', 'Purchase_Amount']
df_sweets_5000 = pd.DataFrame(data_5000, columns=columns)

# Save to CSV
file_path_5000 = r'C:\Users\Rupesh Shinde\Desktop\CRM\sweets_segmentation_data.csv'
df_sweets_5000.to_csv(file_path_5000, index=False)

file_path_5000
