import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google_play_scraper import Sort, reviews
import json
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator, exceptions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time

# Load kamus normalisasi dari file 'slang_words.txt'
# Membaca dan memuat file kamus slang yang berisi kata-kata yang akan dinormalisasi.
with open('slang_words.txt', 'r', encoding='utf-8-sig') as file:
    data = file.read()
    norm = json.loads(data)

# Fungsi untuk menormalisasi teks berdasarkan kamus normalisasi.
def normalize_text(text, normalization_dict):
    words = text.split()
    normalized_words = [normalization_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)

# Fungsi untuk scraping ulasan dari beberapa aplikasi secara bersamaan.
# Dapat menyaring ulasan berdasarkan tanggal dan membatasi jumlah ulasan yang diambil.
def scrape_reviews_multiple_apps(app_ids, app_names, max_reviews=None, sleep_milliseconds=0, start_date=None, end_date=None):
    all_reviews = []
    reviews_count_per_app = {}
    start_date = pd.to_datetime(start_date) if start_date else None
    end_date = pd.to_datetime(end_date) if end_date else None

    # Looping melalui semua aplikasi yang dimasukkan oleh pengguna
    for app_id, app_name in zip(app_ids, app_names):
        st.write(f"Scraping reviews for app: {app_name}")
        result = []
        continuation_token = None

        # Initialize progress bar
        progress_bar = st.progress(0)
        
        try:
            while len(result) < max_reviews:
                try:
                    # Mengambil ulasan dari Google Play Store menggunakan API
                    new_result, continuation_token = reviews(
                        app_id,
                        continuation_token=continuation_token,
                        lang='id',
                        country='id',
                        sort=Sort.NEWEST,
                        filter_score_with=None,
                        count=199
                    )
                    if not new_result:
                        break
                    # Normalisasi teks ulasan
                    for review in new_result:
                        review['content'] = normalize_text(review['content'], norm)

                    # Menyaring ulasan berdasarkan tanggal
                    for review in new_result:
                        review_date = pd.to_datetime(review['at'])
                        if start_date <= review_date <= end_date:
                            result.append(review)
                            if len(result) >= max_reviews:
                                break
                    # Update progress bar
                    progress_bar.progress(min(len(result) / max_reviews, 1.0))
                    time.sleep(0.1)  # Tambahkan delay kecil untuk membiarkan UI memperbarui
                except Exception as e:
                    st.write(f"Error occurred: {e}")
                    time.sleep(5)  # Tunggu 5 detik sebelum mencoba lagi
        except KeyboardInterrupt:
            st.write("Process interrupted by user.")
            break  # Keluar dari loop for jika pengguna menginterupsi

        st.write(f"Total reviews scraped for {app_name}: {len(result)}")
        reviews_count_per_app[app_name] = len(result)

        # Simpan hasil ulasan untuk setiap aplikasi
        result = [{'content': review['content'], 'at': review['at'], 'appName': app_name} for review in result]
        all_reviews.extend(result)

        if sleep_milliseconds:
            time.sleep(sleep_milliseconds / 1000)

    st.write(f"Total reviews scraped before filtering: {len(all_reviews)}")
    progress_bar.progress(1.0)  # Set progress bar to 100% when done

    # Hitung jumlah ulasan per aplikasi setelah filtering
    reviews_count_per_app_filtered = pd.DataFrame(all_reviews)['appName'].value_counts().to_dict()

    return pd.DataFrame(all_reviews), reviews_count_per_app_filtered

# 1. Merubah jenis huruf menjadi huruf kecil
def lowercase(review_text):
    return review_text.lower()

# 2. Menghapus emoji menggunakan regex dan nilai unicode dari emoji
def remove_emoji(review_text):
    emoji_pattern = re.compile(
        "["u"\U00002070"
        u"\U000000B9"
        u"\U000000B2-\U000000B3"
        u"\U00002074-\U00002079"
        u"\U0000207A-\U0000207E"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', review_text)

# 3. Menghapus hashtag
def remove_hashtag(review_text, default_replace=""):
    return re.sub(r'#\w+', default_replace, review_text)

# 4. Menghapus tanda baca
def remove_punctuation(review_text, default_text=" "):
    list_punct = string.punctuation
    delete_punct = str.maketrans(list_punct, ' ' * len(list_punct))
    return ' '.join(review_text.translate(delete_punct).split())

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
    return re.sub(r'(.)\1+', r'\1\1', review_text)

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
    return review_text.split()

# 12. Menghilangkan stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
def remove_stopwords(review_text):
    return [word for word in review_text if word not in stop_words]

# 13. Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem_text(review_text):
    return [stemmer.stem(word) for word in review_text]

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
    review_text = remove_digits(review_text)
    review_text = remove_extra_whitespace(review_text)
    return review_text

# Fungsi penerjemahan teks ke bahasa Inggris menggunakan Google Translator
def translate_text(text, retries=3):
    for i in range(retries):
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        except exceptions.TranslationNotFound as e:
            st.write(f"Translation error: {e}. Retrying... ({i + 1}/{retries})")
            time.sleep(2)
        except Exception as e:
            st.write(f"Error translating text: {e}. Retrying... ({i + 1}/{retries})")
            time.sleep(2)
    return text

# Fungsi untuk menghitung bobot sentimen dan menghasilkan rangking aplikasi
def calculate_sentiment_weight(df):
    # Menghitung jumlah ulasan positif, negatif, dan netral per aplikasi
    sentimen_positif = df[df['Sentiment'] == 'Positif'].groupby('appName')['Sentiment'].count()
    sentimen_negatif = df[df['Sentiment'] == 'Negatif'].groupby('appName')['Sentiment'].count()
    sentimen_netral = df[df['Sentiment'] == 'Netral'].groupby('appName')['Sentiment'].count()

    # Membuat DataFrame untuk menyimpan hasil perhitungan sentimen
    df_sentimen = pd.DataFrame({
        'Positif': sentimen_positif,
        'Negatif': sentimen_negatif,
        'Netral': sentimen_netral
    }).fillna(0)

    # Normalisasi
    normalized_positif = df_sentimen['Positif'] / df_sentimen['Positif'].max()
    normalized_negatif = df_sentimen['Negatif'].min() / df_sentimen['Negatif']
    normalized_netral = (df_sentimen['Netral'].min() / df_sentimen['Netral']) + (df_sentimen['Netral'] / df_sentimen['Netral'].max())
    normalized_netral = normalized_netral / 2

    # Bobot untuk setiap sentimen
    bobot_positif = 0.61
    bobot_negatif = 0.28
    bobot_netral = 0.11

    # Hitung hasil bobot
    hasil_bobot_positif = normalized_positif * bobot_positif
    hasil_bobot_negatif = normalized_negatif * bobot_negatif
    hasil_bobot_netral = normalized_netral * bobot_netral
    jumlah = hasil_bobot_positif.add(hasil_bobot_negatif, fill_value=0).add(hasil_bobot_netral, fill_value=0)
    df_result = pd.DataFrame(jumlah, columns=['Nilai Preferensi']).reset_index()
    df_result = df_result.rename(columns={'index': 'appName'})
    df_result = df_result.sort_values(by='Nilai Preferensi', ascending=False).reset_index(drop=True)
    return df_result

# Fungsi untuk menggambar diagram batang sentimen
def diagram_batang1(df):
    plt.figure(figsize=(10, 6))
    palette = {'Positif': 'green', 'Netral': 'blue', 'Negatif': 'red'}
    sns.countplot(data=df, x='appName', hue='Sentiment', palette=palette)
    plt.title('Sentimen Tiap Aplikasi', fontweight='bold')
    plt.xlabel('Aplikasi')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    st.pyplot(plt)

# Fungsi untuk menggambar diagram batang sentimen 
def diagram_batang2(df):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("bright")
    sns.countplot(data=df, x='Sentiment', hue='appName', palette=palette)
    plt.title('Sentimen Tiap Aplikasi', fontweight='bold')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    st.pyplot(plt)

# Fungsi utama untuk menjalankan seluruh proses scraping, preprocessing, analisis sentimen, dan perhitungan bobot
st.title("Perangkingan Aplikasi Google Play Store")
# Mengambil input dari pengguna
num_apps = st.number_input("Masukkan jumlah aplikasi yang ingin diambil ulasannya", min_value=1, value=1)

app_ids = []
app_names = []
for i in range(num_apps):
    col1, col2 = st.columns(2)
    with col1:
        app_id = st.text_input(f"Masukkan ID aplikasi ke-{i+1} (contoh: com.gojek.gopay):")
    with col2:
        app_name = st.text_input(f"Masukkan nama aplikasi ke-{i+1}:")
    app_ids.append(app_id)
    app_names.append(app_name)

max_reviews = st.number_input("Masukkan jumlah maksimum ulasan yang ingin diambil", min_value=1, value=10)
sleep_milliseconds = 3

# Tambahkan input rentang waktu
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Tanggal mulai", value=datetime.now() - timedelta(days=500))
with col2:
    end_date = st.date_input("Tanggal akhir", value=datetime.now())

# input terjemahan
dest_language = st.text_input("Masukkan bahasa tujuan untuk terjemahan (contoh: 'en' untuk Inggris):")

if st.button("Mulai Pengambilan Ulasan"):
    all_reviews, reviews_count_per_app = scrape_reviews_multiple_apps(app_ids, app_names, max_reviews, sleep_milliseconds, start_date, end_date)
    st.write(f"Jumlah ulasan yang dikumpulkan: {len(all_reviews)}")
    
    # Membuat DataFrame dari daftar ulasan yang sudah diambil, hanya mengambil kolom 'at', 'appName', dan 'content'.
    df = pd.DataFrame(all_reviews)[['at', 'appName','content']]
    # Membersihkan teks ulasan menggunakan fungsi preprocessing yang telah didefinisikan sebelumnya
    df['clean_review'] = df['content'].apply(clean_review_pipeline)
    # Melakukan tokenisasi pada ulasan yang sudah dibersihkan
    df['tokenize'] = df['clean_review'].apply(tokenize)
    # Menghapus stopwords dari teks yang telah ditokenisasi
    df['filtering'] = df['tokenize'].apply(remove_stopwords)
    # Melakukan stemming pada teks yang sudah difilter, menggabungkan kata-kata hasil stemming menjadi string
    df['stemming_data'] = df['filtering'].apply(lambda x: ' '.join(stem_text(x)))
    
    # Jika destinasi bahasa diatur, terjemahkan teks hasil stemming ke bahasa Inggris
    if dest_language:
        df['english'] = df['stemming_data'].apply(lambda x: translate_text(x, retries=3))
    else:
        # Jika tidak, gunakan teks hasil stemming seperti adanya
        df['english'] = df['stemming_data']
    
    # Melakukan analisis sentimen menggunakan VADER
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(x) for x in df['english']]
    df['Compound_Score'] = [x['compound'] for x in scores]
    
    # Mengklasifikasikan sentimen berdasarkan compound score
    df.loc[df['Compound_Score'] >= 0.05, 'Sentiment'] = 'Positif'
    df.loc[(df['Compound_Score'] > -0.05) & (df['Compound_Score'] < 0.05), 'Sentiment'] = 'Netral'
    df.loc[df['Compound_Score'] <= -0.05, 'Sentiment'] = 'Negatif'
    
    # Menghitung bobot sentimen untuk setiap aplikasi dan menghasilkan hasil akhir
    df_result = calculate_sentiment_weight(df)
    st.write(df_result)
    
    st.write(df)
    # Menampilkan diagram batang untuk visualisasi data sentimen (fungsi harus didefinisikan di tempat lain)
    diagram_batang1(df)
    diagram_batang2(df)
    
    # Menghitung jumlah sentimen untuk setiap aplikasi
    sentimen_count = df.groupby(['appName', 'Sentiment']).size().unstack(fill_value=0)
    st.write(sentimen_count)
    
    # Menghitung total ulasan per aplikasi
    total_reviews_per_app = df['appName'].value_counts().reset_index()
    total_reviews_per_app.columns = ['appName', 'Total Reviews']
    st.write(total_reviews_per_app)
    
    # Mengecek nilai yang hilang sebelum dan sesudah
    missing_values_sebelum = df.isnull().sum()
    df = df.dropna()
    missing_values_sesudah = df.isnull().sum()
    
    # Menampilkan total jumlah ulasan yang tersisa setelah missing_value dan pembersihan data ulasan 
    st.write(f"Total semua ulasan: {len(df)}")
    
