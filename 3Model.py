import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = r'C:\Users\Rupesh Shinde\Desktop\CRM\synthetic.csv'  # Replace with actual path if it's different
data = pd.read_csv(file_path)

# Display first few rows of the dataset for reference
print(data.head())

# Convert "Sentiment" to numerical (optional for use in models)
sentiment_mapping = {"Positive": 1, "Negative": 0}
data['Sentiment_Numeric'] = data['Sentiment'].map(sentiment_mapping)

# Pivot table for collaborative filtering
pivot_table = data.pivot_table(index='CustomerID', columns='Sweet', values='Rating').fillna(0)

# Apply SVD to the pivot table for dimensionality reduction
svd = TruncatedSVD(n_components=2)
matrix = svd.fit_transform(pivot_table)

# Compute cosine similarity between customer vectors
similarity_matrix = cosine_similarity(matrix)

def recommend_sweets(customer_id, top_n=3):
    """
    Recommend top N sweets for a given customer based on collaborative filtering.
    """
    customer_index = pivot_table.index.get_loc(customer_id)
    customer_ratings = similarity_matrix[customer_index]
    similar_customers = sorted(enumerate(customer_ratings), key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    recommended_sweets = []
    for index, _ in similar_customers:
        recommended_sweets += pivot_table.iloc[index].sort_values(ascending=False).index.tolist()
    
    return list(set(recommended_sweets))

# Example usage: Get customer ID from user input
customer_id = int(input("Enter CustomerID for recommendations: "))  # Change this to an actual CustomerID from your dataset
recommendations = recommend_sweets(customer_id, top_n=3)
print(f"Recommendations for Customer {customer_id}: {recommendations}")

# Region-Specific Campaign Generation
region_campaigns = {
    "North India": {
        "Language": "Hindi",
        "Message": "उत्तर भारत में दीवाली के इस त्यौहार पर स्वादिष्ट गुलाब जामुन और लड्डू का आनंद लें!"
    },
    "South India": {
        "Language": "Tamil",
        "Message": "தீபாவளி பண்டிகைக்கு தென்னிந்தியாவில் சுவையான மைசூர் பாக் மற்றும் கொழுக்கட்டை உண்டுபாருங்கள்!"
    },
    "East India": {
        "Language": "Bengali",
        "Message": "পূর্ব ভারতে দীপাবলির সময় সুস্বাদু সন্দেশ এবং পিঠা উপভোগ করুন!"
    }
}

def generate_campaign(region):
    """
    Generate a region-specific marketing campaign message.
    """
    campaign = region_campaigns.get(region, {
        "Language": "English",
        "Message": "Enjoy the festive season with our special sweets!"
    })
    return campaign

# Example usage: Get region from user input
region = input("Enter region for campaign generation (North India, South India, East India): ")
campaign = generate_campaign(region)
print(f"Campaign Message in {campaign['Language']}: {campaign['Message']}")

# Cultural Context Mapping
cultural_events = {
    "Ganesh Chaturthi": ["Modak", "Ladoo"],
    "Diwali": ["Kaju Katli", "Rasgulla", "Gulab Jamun"],
    "Onam": ["Payasam", "Mysore Pak"],
    "Baisakhi": ["Jalebi", "Imarti"],
    "Pongal": ["Pongal Sweet", "Kozhukattai"]
}

def map_cultural_event_to_sweets(event_name):
    """
    Map a cultural event to its associated sweets.
    """
    return cultural_events.get(event_name, ["Generic Sweets"])

# Example usage: Get cultural event from user input
event_name = input("Enter a cultural event name (e.g., Ganesh Chaturthi, Diwali): ")
mapped_sweets = map_cultural_event_to_sweets(event_name)
print(f"Sweets for {event_name}: {mapped_sweets}")
