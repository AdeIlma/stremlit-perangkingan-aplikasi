import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator, exceptions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st
from google_play_scraper import Sort, reviews
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Download data yang diperlukan untuk tokenisasi dan stopwords (jalankan hanya sekali)
nltk.download('punkt')
nltk.download('stopwords')



# 1. Merubah jenis huruf menjadi huruf kecil
def lowercase(review_text):
    return review_text.lower()

# 2. Menghapus emoji menggunakan regex dan nilai unicode dari emoji
def remove_emoji(review_text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', review_text)

# 3. Menghapus hashtag
def remove_hashtag(review_text, default_replace=""):
    return re.sub(r'#\w+', default_replace, review_text)

# 4. Menghapus tanda baca
def remove_punctuation(review_text, default_text=" "):
    return re.sub(r'[^\w\s]', default_text, review_text)

# 5. Menghapus superscript
def remove_superscript(review_text):
    superscript_pattern = re.compile(
        "["u"\U00002070"
        u"\U000000B9"
        u"\U000000B2-\U000000B3"
        u"\U00002074-\U00002079"
        u"\U0000207A-\U0000207E"
        "]+", flags=re.UNICODE)
    return superscript_pattern.sub(r'', review_text)

# 6. Membatasi pengulangan huruf
def word_repetition(review_text):
    return re.sub(r'(.)\1{1,}', r'\1\1', review_text)

# 7. Membatasi pengulangan kata
def repetition(review_text):
    return re.sub(r'\b(\w+)(?:\W\1\b)+', r'\1', review_text, flags=re.IGNORECASE)

# 8. Menghapus URL
def remove_URL(review_text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', review_text)

# 9. Menghapus angka
def remove_digits(review_text):
    return re.sub(r'\d+', '', review_text)

# 10. Menghapus spasi berlebih
def remove_extra_whitespace(review_text):
    return ' '.join(review_text.split())

# 11. Proses tokenisasi
def tokenize(review_text):
    return word_tokenize(review_text)  

# 12. Menghilangkan stopwords
stop_words = set(stopwords.words('indonesian'))
def remove_stopwords(review_text):
    return [word for word in review_text if word not in stop_words]

# 13. Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem_text(review_text):
    return [stemmer.stem(word) for word in review_text]

def list_to_string(text_list):
    return ' '.join(text_list)

# Menggabungkan semua fungsi ke dalam pipeline
def clean_review_pipeline(review_text):
    review_text = lowercase(review_text)
    review_text = remove_emoji(review_text)
    review_text = remove_hashtag(review_text)
    review_text = remove_punctuation(review_text)
    review_text = remove_superscript(review_text)
    review_text = word_repetition(review_text)
    review_text = repetition(review_text)
    review_text = remove_URL(review_text)
    review_text = tokenize(review_text)
    review_text = ' '.join(review_text)  # Mengonversi list ke string
    review_text = remove_digits(review_text)
    review_text = remove_extra_whitespace(review_text)
    
        # Check if the review text is empty after preprocessing
    if review_text.strip() == '':
        return None  # Return None for empty reviews
    return review_text

# Function to translate text using Google Translator with retry mechanism
def translate_text(text, dest_language, retries=3):
    for i in range(retries):
        try:
            translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
            return translated
        except exceptions.TranslationNotFound as e:
            print(f"Translation error: {e}. Retrying... ({i + 1}/{retries})")
            time.sleep(2)
        except Exception as e:
            print(f"Error translating text: {e}. Retrying... ({i + 1}/{retries})")
            time.sleep(2)
    return text

# Function to perform Vader sentiment analysis
def vader_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

# Function to calculate sentiment weight
def calculate_sentiment_weight(df):
    # Normalisasi sentimen
    df.loc[df['Compound_Score'] < 0, 'Sentiment'] = 'Negatif'
    df.loc[df['Compound_Score'] == 0, 'Sentiment'] = 'Netral'
    df.loc[df['Compound_Score'] > 0, 'Sentiment'] = 'Positif'

    # Menghitung jumlah sentimen per aplikasi
    sentimen_positif = df[df['Sentiment'] == 'Positif'].groupby('appName')['Sentiment'].count()
    sentimen_negatif = df[df['Sentiment'] == 'Negatif'].groupby('appName')['Sentiment'].count()
    sentimen_netral = df[df['Sentiment'] == 'Netral'].groupby('appName')['Sentiment'].count()


    # Menggabungkan semua sentimen dalam satu DataFrame dan mengisi NaN dengan 0
    df_sentimen = pd.DataFrame({
        'Positif': sentimen_positif,
        'Negatif': sentimen_negatif,
        'Netral': sentimen_netral
    }).fillna(0)

    # Normalisasi sentimen
    normalized_positif = df_sentimen['Positif'] / df_sentimen['Positif'].max()
    normalized_negatif = df_sentimen['Negatif'].min() / df_sentimen['Negatif']
    normalized_netral = (df_sentimen['Netral'].min() / df_sentimen['Netral']) + (df_sentimen['Netral'] / df_sentimen['Netral'].max())
    normalized_netral = normalized_netral / 2

    # Bobot sentimen
    bobot_positif = 0.61
    bobot_negatif = 0.11
    bobot_netral = 0.28

    # Menghitung bobot sentimen
    hasil_bobot_positif = normalized_positif * bobot_positif
    hasil_bobot_negatif = normalized_negatif * bobot_negatif
    hasil_bobot_netral = normalized_netral * bobot_netral

    # Menghitung total bobot sentimen untuk setiap aplikasi
    jumlah = hasil_bobot_positif.add(hasil_bobot_negatif, fill_value=0).add(hasil_bobot_netral, fill_value=0)

    # Mengubah hasil menjadi DataFrame
    df_result = pd.DataFrame(jumlah, columns=['Bobot']).reset_index()
    df_result = df_result.rename(columns={'index': 'Aplikasi'})

    # Mengurutkan DataFrame berdasarkan nilai bobot total dari yang terbesar
    df_result = df_result.sort_values(by='Bobot', ascending=False).reset_index(drop=True)

    return df_result

# Function to plot sentiment for each app
def plot_sentiment_diagram_batang(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='appName', hue='Sentiment')
    plt.title('Sentimen Tiap Aplikasi', fontweight='bold')  # Make the title bold
    plt.xlabel('Aplikasi')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')  # Menambahkan judul pada legenda
    st.pyplot(plt)

def plot_sentiment_diagram_garis(df):
    # Agregasi data untuk mendapatkan jumlah ulasan per aplikasi berdasarkan sentimen
    df_aggregated = df.groupby(['appName', 'Sentiment']).size().reset_index(name='count')

    # Pastikan data diurutkan berdasarkan aplikasi dan sentimen
    df_aggregated = df_aggregated.sort_values(by=['appName', 'Sentiment'])
    plt.figure(figsize=(10, 6))
    palette = {'Positif': 'green', 'Netral': 'blue', 'Negatif': 'red'}  # Warna kustom
    sns.lineplot(data=df_aggregated, x='appName', y='count', hue='Sentiment', palette=palette, marker='o')
    plt.title('Sentimen Tiap Aplikasi', fontweight='bold')  # Membuat judul tebal
    plt.xlabel('Aplikasi')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')  # Menambahkan judul pada legenda
    st.pyplot(plt)

# Main function
def main():
    st.title("Perangkingan Aplikasi Menggunakan  ulasan Google Play Store")

    num_apps = st.number_input("Masukkan jumlah aplikasi yang ingin diambil ulasannya", min_value=1, value=1)

    app_ids = []
    app_names = []
    for i in range(num_apps):
        app_id = st.text_input(f"Masukkan ID aplikasi Google Play Store ke-{i+1} (contoh: com.gojek.gopay):")
        app_name = st.text_input(f"Masukkan nama aplikasi ke-{i+1}:")
        app_ids.append(app_id)
        app_names.append(app_name)

    max_reviews = st.number_input("Masukkan jumlah maksimum ulasan yang ingin diambil", min_value=1, value=1000)
    sleep_milliseconds = st.number_input("Masukkan waktu jeda antara setiap permintaan scraping (dalam milidetik)", min_value=0, value=0)

    translate_reviews = st.checkbox("Terjemahkan ke bahasa lain?")
    dest_language = None
    if translate_reviews:
        dest_language = st.text_input("Masukkan kode bahasa tujuan (contoh: en untuk bahasa Inggris):")

    if st.button("Mulai Pengambilan Ulasan"):
        reviews_data = []
        for app_id, app_name in zip(app_ids, app_names):
            print(f"Scraping reviews for app: {app_name}")
            result = []
            continuation_token = None

            with tqdm(total=max_reviews, position=0, leave=True) as pbar:
                while len(result) < max_reviews:
                    try:
                        new_result, continuation_token = reviews(
                            app_id,
                            continuation_token=continuation_token,
                            lang='id',
                            country='id',
                            sort=Sort.NEWEST,
                            filter_score_with=None,
                            count=min(max_reviews - len(result), 199)  # Adjust count based on remaining reviews needed
                        )
                        if not new_result:
                            break
                        result.extend(new_result)
                        pbar.update(len(new_result))
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        time.sleep(5)

            result = [{'content': review['content'], 'appName': app_name} for review in result]
            reviews_data.extend(result)

            if sleep_milliseconds:
                time.sleep(sleep_milliseconds / 1000)

        # Create DataFrame from reviews data
        df = pd.DataFrame(reviews_data)

        # Menghapus data duplikat dan menangani missing value sebelum preprocessing
        df.drop_duplicates(inplace=True)  # Menghapus data duplikat
        # Drop rows with empty reviews after preprocessing
        df['clean_review'] = df['content'].apply(clean_review_pipeline)
        df.dropna(subset=['clean_review'], inplace=True)


        # Membersihkan teks dan membuat kolom 'clean_review'
        df['clean_review'] = df['content'].apply(clean_review_pipeline)

        # Lakukan tokenisasi pada data frame
        df['tokenize'] = df['clean_review'].apply(lambda x: tokenize(x))  
        # Menghilangkan stopwords
        df['filtering'] = df['tokenize'].apply(lambda x: remove_stopwords(x))
        # Stemming
        df['stemming_data'] = df['filtering'].apply(lambda x: ' '.join(stem_text(x)))

        # Memeriksa nilai kosong dan mengisinya dengan string kosong jika ada
        df['stemming_data'].fillna('', inplace=True)

        # Translation
        if translate_reviews and dest_language:
            df['translate_reviews'] = df['stemming_data'].apply(translate_text, args=(dest_language,))

        # Perform Vader sentiment analysis
        df['Compound_Score'] = df['translate_reviews'].apply(vader_sentiment_analysis)
        # Add Sentiment column to the original DataFrame
        df['Sentiment'] = df['Compound_Score'].apply(lambda score: 'Positif' if score > 0 else ('Netral' if score == 0 else 'Negatif'))

        st.write(df)

        # Calculate sentiment weight
        df_result = calculate_sentiment_weight(df)
        st.write(df_result)

        # Plot sentiment for each app
        plot_sentiment_diagram_batang(df)
        plot_sentiment_diagram_garis(df)

if __name__ == "__main__":
    main()

