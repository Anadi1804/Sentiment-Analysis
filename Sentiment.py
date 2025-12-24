import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import io
import re
import emoji
from googleapiclient.discovery import build
import string
from collections import Counter

# Constants and Configurations
YOUTUBE_API_KEY = "AIzaSyCew87ax6ttughaD5hTRDruV8h_rZCaIKU"  # Replace with your API key

# Initialize BERT model with caching
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None

sentiment_pipeline = load_sentiment_model()
if sentiment_pipeline is None:
    st.error("Failed to load sentiment analysis model. Please try again.")
    st.stop()

# Dictionaries
slang_dict = {
    "u": "you",
    "r": "are",
    "b4": "before",
    "lmk": "let me know",
    "btw": "by the way",
    "idk": "i don't know",
    "imo": "in my opinion",
    "fyi": "for your information"
}

emoji_sentiment_dict = {
    "😊": "positive",
    "😃": "positive",
    "😄": "positive",
    "👍": "positive",
    "❤": "positive",
    "✨": "positive",
    "😢": "negative",
    "😠": "negative",
    "😭": "negative",
    "👎": "negative",
    "🤔": "neutral",
    "😐": "neutral"
}

def clean_text(text):
    """Basic text cleaning without NLTK dependency"""
    try:
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    except Exception as e:
        st.error(f"Error in text cleaning: {str(e)}")
        return text

def handle_emojis(text):
    """Convert emojis to sentiment"""
    sentiment = None
    for emoji_char, emoji_sentiment in emoji_sentiment_dict.items():
        if emoji_char in text:
            sentiment = emoji_sentiment
            break
    return text, sentiment

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    try:
        if not isinstance(text, str):
            return str(text), None
            
        # Handle emojis first
        text, emoji_sentiment = handle_emojis(text)
        
        if emoji_sentiment:
            return text, emoji_sentiment
            
        # Clean text
        text = clean_text(text)
        
        # Replace slang
        for slang, meaning in slang_dict.items():
            text = text.replace(slang, meaning)
        
        return text, None
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text, None

def analyze_sentiment(text):
    """Analyze sentiment of given text"""
    try:
        processed_text, emoji_sentiment = preprocess_text(text)
        
        if emoji_sentiment:
            return emoji_sentiment, 1.0
            
        if not processed_text.strip():
            return 'neutral', 0.5
            
        result = sentiment_pipeline(processed_text[:512])[0]
        label = result['label']
        score = result['score']
        
        if label == 'LABEL_2':
            return 'positive', score
        elif label == 'LABEL_0':
            return 'negative', score
        else:
            return 'neutral', score
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return 'neutral', 0.0

def plot_sentiment_pie_chart(df):
    """Create pie chart of sentiment distribution"""
    try:
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        patches, texts, autotexts = ax.pie(
            sentiment_counts.values,
            colors=colors[:len(sentiment_counts)],
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        
        plt.axis('equal')
        return fig
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        return None

def get_youtube_comments(video_url):
    """Fetch comments from YouTube video"""
    try:
        if 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]
        elif 'youtube.com/watch?v=' in video_url:
            video_id = video_url.split('v=')[1].split('&')[0]
        else:
            raise ValueError("Invalid YouTube URL format")
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        comments = []
        next_page_token = None
        
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token or len(comments) >= 500:  # Limit to 500 comments
                break
        
        return comments
    except Exception as e:
        st.error(f"Error fetching YouTube comments: {str(e)}")
        return []

def analyze_csv(file):
    """Analyze sentiment from CSV file"""
    try:
        df = pd.read_csv(file)
        if len(df.columns) > 1:
            st.warning("Multiple columns detected. Using first column for analysis.")
        
        # Use first column for analysis
        text_column = df.iloc[:, 0]
        df = pd.DataFrame({'text': text_column})
        df = df.dropna(subset=['text'])
        
        if df.empty:
            st.warning("No valid text found in CSV file.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None

def main():
    print("Script is running...")
    st.title("Sentiment Analysis")
    st.write("Analyze sentiment from text, CSV, or YouTube comments")
    
    # Sidebar for input selection
    analysis_type = st.sidebar.radio(
        "Choose input type:",
        ["Text Input", "CSV Upload", "YouTube URL"]
    )
    
    if analysis_type == "Text Input":
        text_input = st.text_area("Enter text to analyze:", height=150)
        if st.button("Analyze Text"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    df = pd.DataFrame({'text': [text_input]})
                    df['sentiment'], df['confidence'] = zip(*df['text'].apply(analyze_sentiment))
                    
                    st.subheader("Results:")
                    st.dataframe(df[['text', 'sentiment', 'confidence']])
                    
                    fig = plot_sentiment_pie_chart(df)
                    if fig:
                        st.pyplot(fig)
            else:
                st.warning("Please enter some text to analyze.")
                
    elif analysis_type == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            with st.spinner("Processing CSV file..."):
                df = analyze_csv(uploaded_file)
                if df is not None:
                    df['sentiment'], df['confidence'] = zip(*df['text'].apply(analyze_sentiment))
                    
                    st.subheader("Results:")
                    st.dataframe(df[['text', 'sentiment', 'confidence']])
                    
                    fig = plot_sentiment_pie_chart(df)
                    if fig:
                        st.pyplot(fig)
                        
    else:  # YouTube URL
        url_input = st.text_input("Enter YouTube URL:")
        if st.button("Analyze Comments"):
            if url_input:
                with st.spinner("Fetching and analyzing YouTube comments..."):
                    comments = get_youtube_comments(url_input)
                    if comments:
                        df = pd.DataFrame({'text': comments})
                        df['sentiment'], df['confidence'] = zip(*df['text'].apply(analyze_sentiment))
                        
                        st.subheader("Results:")
                        st.dataframe(df[['text', 'sentiment', 'confidence']])
                        
                        fig = plot_sentiment_pie_chart(df)
                        if fig:
                            st.pyplot(fig)
                    else:
                        st.warning("No comments found or error occurred while fetching comments.")
            else:
                st.warning("Please enter a YouTube URL.")

if __name__ == "__main__":
    main()