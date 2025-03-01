import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
# Load the data
file_path = 'amazon_alexa.tsv'
data = pd.read_csv(file_path, sep='\t')

# Drop rows with null verified_reviews
data = data.dropna(subset=['verified_reviews'])

# Function to perform sentiment analysis
def get_sentiment(text):
    if not isinstance(text, str):  # Check if input is a string
        return 'Unknown'  # Handle non-string input
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to the verified reviews
data['Sentiment'] = data['verified_reviews'].apply(get_sentiment)



# Streamlit app
st.title('Amazon Alexa Product Sentiment Analysis')

# Display data
st.write("### Amazon Alexa Reviews Data")
st.dataframe(data)

# Show sentiment distribution
st.write("### Sentiment Distribution")
sentiment_counts = data['Sentiment'].value_counts()

# Plot the sentiment distribution
fig, ax = plt.subplots()
sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
st.pyplot(fig)

# Allow user to filter by sentiment
sentiment_filter = st.selectbox('Filter reviews by sentiment:', ['All', 'Positive', 'Negative', 'Neutral'])

if sentiment_filter != 'All':
    filtered_data = data[data['Sentiment'] == sentiment_filter]
else:
    filtered_data = data

# Display filtered reviews
st.write(f"### Reviews ({sentiment_filter})")
st.dataframe(filtered_data[['verified_reviews', 'Sentiment']])
