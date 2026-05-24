import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def batch_clean(texts: list) -> list:
    return [clean_text(t) for t in texts]


def get_top_keywords(texts: list, n: int = 20) -> list:
    from collections import Counter
    all_tokens = []
    for text in texts:
        all_tokens.extend(clean_text(text).split())
    return Counter(all_tokens).most_common(n)


if __name__ == '__main__':
    sample = "BREAKING: Scientists EXPOSED for hiding SECRET climate data!! Click here NOW!"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
