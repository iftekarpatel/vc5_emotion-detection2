import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any, Callable, Optional
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    filename="logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error("Error in lemmatization: %s", e)
        return text

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logging.error("Error removing stop words: %s", e)
        return text

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error("Error removing numbers: %s", e)
        return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logging.error("Error converting to lower case: %s", e)
        return text

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error("Error removing punctuations: %s", e)
        return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error("Error removing URLs: %s", e)
        return text

def remove_small_sentences(df: pd.DataFrame) -> None:
    """Set text to NaN if sentence has fewer than 3 words."""
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        logging.info("Small sentences removed from DataFrame")
    except Exception as e:
        logging.error("Error removing small sentences: %s", e)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed")
        return df
    except Exception as e:
        logging.error("Error normalizing text: %s", e)
        raise

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error("Error normalizing sentence: %s", e)
        return sentence

def load_data(path: str) -> Optional[pd.DataFrame]:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info("Data loaded from %s", path)
        return df
    except Exception as e:
        logging.error("Error loading data from %s: %s", path, e)
        return None

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info("Data saved to %s", path)
    except Exception as e:
        logging.error("Error saving data to %s: %s", path, e)
        raise

def main() -> None:
    """Main function to orchestrate data preprocessing."""
    try:
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")
        if train_data is None or test_data is None:
            raise ValueError("Train or test data could not be loaded.")
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing pipeline completed successfully")
    except Exception as e:
        logging.error("Data preprocessing pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()