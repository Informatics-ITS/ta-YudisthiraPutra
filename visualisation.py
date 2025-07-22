
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from collections import Counter
import matplotlib.pyplot as plt
import stanza
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn, stopwords
from nltk.corpus import wordnet as wn
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time
import ast

# Load the saved model
loaded_model = joblib.load('../data/best_model_one_svm_exp_tf.joblib')
pos_path = "../InSet-master/positive.tsv"
neg_path = "../InSet-master/negative.tsv"

# Title of the app

# Set page config to full screen
st.set_page_config(
    page_title="Character Extraction and Clustering",
    layout="wide"  # 'wide' makes it full-width
)

# Title
st.title("Character Extraction and Clustering")

# Subtitle / label before text input
st.subheader("Insert Indonesian Folklore")

# Text input
user_input = st.text_input("Folklore Text...",)
progress = st.progress(0)
status_text = st.empty()

# CLEANING
def clean_text(text):
    # Change tabs into space
    text = text.replace('\t', ' ')
    
    # Replace double double-quotes ("") with a single double-quote (")
    text = text.replace('""', '"')

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    # Remove the first and last double quote if they exist
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]  # Remove first and last characters
    
    # Return the cleaned text
    return text

def standarize_quotation_text(input_text):
    # Step 1 - Standardize quotation marks
    quotation_marks = ['"', "“", "”", "''", "``", "’’", "’", '"""']
    for mark in quotation_marks:
        input_text = input_text.replace(mark, '"')

    # Step 2 - Clean the text (assuming clean_text is defined)
    input_text = clean_text(input_text)

    return input_text

cleaned_text = standarize_quotation_text(user_input)
sentences = sent_tokenize(cleaned_text)
tokens = word_tokenize(cleaned_text)

if cleaned_text:
    progress.progress(20)
    status_text.text(f"Text Cleaned, continue to extract characters.")
    time.sleep(1)

# CHARACTER EXTRACTION
# CAPITAL TOKEN EXTRACTION
def capital_token_extraction(narrative_sentences):
    upper_token = []
    stopwords_id = set(stopwords.words('indonesian'))  # Faster lookup

    for sentence in narrative_sentences:
        if '"' in sentence:
            continue  # Skip quoted sentences

        tokens = word_tokenize(sentence)
        if not tokens:
            continue
        tokens[0] = tokens[0].lower()  # Lowercase first token

        i = 0
        while i < len(tokens):
            token = tokens[i]
            initial_token = i

            if token[0].isupper():
                combined_token = token

                # Concatenate adjacent capitalized tokens
                while i + 1 < len(tokens) and tokens[i + 1][0].isupper():
                    i += 1
                    combined_token += " " + tokens[i]

                # Logic for multi-token names not at the start
                if initial_token - 1 == 0:
                    words = combined_token.split()
                    if len(words) > 1 and not any(word.lower() in stopwords_id for word in words):
                        upper_token.append(combined_token)
                else:
                    if combined_token.lower() not in stopwords_id:
                        upper_token.append(combined_token)

            i += 1

    # Remove duplicates while preserving order
    filtered_tokens = list(dict.fromkeys(upper_token))
    return filtered_tokens

capital_token_extraction_result = capital_token_extraction(sentences)

# SYNSET EXTRACTION

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def extract_synset_from_tokens(tokens):
    include_if = {"living_thing.n.01", "peoples.n.01", "parent.n.01", "relative.n.01", "animal.n.01"}
    must_have_any = {"person.n.01", "people.n.01", "animal.n.01"}  # Accept either one
    result = []

    # Use set for uniqueness
    tokens = list(set(tokens))
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    for word in stemmed_tokens:
        synsets = wn.synsets(word, lang='ind')
        filtered = [syn for syn in synsets if "United States" not in syn.definition()]

        for syn in filtered[:3]:  # Limit check to top 3 synsets
            for path in syn.hypernym_paths():
                hypernym_names = {h.name() for h in path}
                if hypernym_names & must_have_any and hypernym_names & include_if:
                    result.append(word)
                    break
            else:
                continue
            break

    return list(set(word.lower() for word in result))

synset_extraction_result = extract_synset_from_tokens(tokens)

def aggregate_candidates(synset_result, capital_tokens):
    # Normalize all to lowercase and merge, remove duplicates
    combined = set([token.lower() for token in synset_result + capital_tokens])
    return list(combined)

candidate_tokens = aggregate_candidates(synset_extraction_result, capital_token_extraction_result)
# S1 EXTRACTION
import time
import stanza

# Load the Indonesian model (only run once)
nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', use_gpu=False)

# Extract full name based on dependencies
def get_full_name_chain(word, sentence):
    name_parts = [word.text]
    current_word = word
    while True:
        next_word = next(
            (w for w in sentence.words if w.head == current_word.id and w.deprel in ['flat:name']),
            None
        )
        if next_word is None:
            break
        if next_word.deprel == 'nmod:poss':
            name_parts.append(next_word.text)
            return ''.join(name_parts)  # or use ' '.join(name_parts)
        else:
            name_parts.append(next_word.text)
        current_word = next_word
    return ' '.join(name_parts)


# Process list of sentences
def extract_subjects_from_sentences(sentences, batch_size=10):
    all_filtered_subjects = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        while True:
            try:
                for sentence_text in batch:
                    full_subjects = []
                    doc = nlp(sentence_text)

                    for sentence in doc.sentences:
                        for word in sentence.words:
                            if word.deprel == 'nsubj':
                                full_name = get_full_name_chain(word, sentence)
                                full_subjects.append(full_name)

                    full_subjects_lower = [s.lower() for s in full_subjects]
                    all_filtered_subjects.extend(full_subjects_lower)

                break  # success
            except Exception as e:
                print(f"Error processing batch: {e}. Retrying...")
                time.sleep(5)

    return all_filtered_subjects

subject_sentences = extract_subjects_from_sentences(sentences, batch_size=10)

def create_subject_match_list(candidate_tokens, subject_list):
    # Lowercase all candidate tokens
    candidate = [token.lower() for token in candidate_tokens]

    final_list = []
    final_list_2 = []

    for subject in subject_list:
        if subject in candidate:
            final_list.append(subject)

    for name in candidate:
        for subject in subject_list:
            if name in subject:
                final_list_2.append(name)
    
    return list(set(final_list_2))

s1_list = create_subject_match_list(candidate_tokens, subject_sentences)

# S2 EXTRACTION
import re

def extract_dialogue_tag_from_sentences(sentences):
    dialogue_tag_list = []
    for sentence in sentences:
        # Remove anything inside double quotes (i.e., dialogue content)
        outside_quotation = re.sub(r'"[^"]*"', '', sentence).strip()
        if outside_quotation:
            dialogue_tag_list.append(outside_quotation)
    return dialogue_tag_list

nltk.download('punkt')
nltk.download('stopwords')

stopwords_id = set(stopwords.words('indonesian'))

def remove_stopwords_from_dialogue(dialogue_tag_list, stopwords):
    return [
        tag for tag in dialogue_tag_list
        if all(word.lower() not in stopwords for word in word_tokenize(tag))
    ]

def extract_subject_from_dialogue(dialogue_tag_list, candidate):
    subject_list = []

    for name in candidate:
        for tag in dialogue_tag_list:
            tokens = word_tokenize(tag)
            for token in tokens:
                if token.lower().startswith(name.lower()):
                    subject_list.append(name.lower())
    return list(set(subject_list))

def subject_from_sentences(sentences, candidate):
    # 1. Extract non-dialogue parts
    dialogue_tags = extract_dialogue_tag_from_sentences(sentences)

    # 2. Remove stopwords
    cleaned_dialogue_tags = remove_stopwords_from_dialogue(dialogue_tags, stopwords_id)

    # 3. Match subject candidates
    matched_subjects = extract_subject_from_dialogue(cleaned_dialogue_tags, candidate)

    return matched_subjects

s2_list = subject_from_sentences(sentences, candidate_tokens)

# S3 EXTRACTION
import re

def extract_connected_words(text, candidate):
    conjunctions = ['dan', 'serta', 'atau', 'tetapi']
    connected = set()

    for i in range(len(candidate)):
        for j in range(len(candidate)):
            if i == j:
                continue
            for conj in conjunctions:
                # Build regex pattern: <candidate1> <conjunction> <candidate2>
                pattern = rf'\b{re.escape(candidate[i])}\b\s+{conj}\s+\b{re.escape(candidate[j])}\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    connected.add(candidate[i])
                    connected.add(candidate[j])

    return list(connected)
s3_list = extract_connected_words(cleaned_text, candidate_tokens)

final_candidate = list(set(
    [item.lower() for item in s1_list + s2_list + s3_list]
))

# EXTRACT POSSESSIVE
import re

def extract_possessives(text, candidates):
    possessive_suffixes = ['nya', 'ku', 'mu']
    seen_lower = {}
    result = []

    # Possessive pattern: candidate + suffix
    for cand in candidates:
        for suffix in possessive_suffixes:
            pattern = rf'\b{re.escape(cand)}{suffix}\b'
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                key = match.lower()
                if key not in seen_lower:
                    seen_lower[key] = match

    # Adjacent names: cand1 cand2
    for cand1 in candidates:
        for cand2 in candidates:
            if cand1 != cand2:
                pattern = rf'\b{re.escape(cand1)} {re.escape(cand2)}\b'
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                for match in matches:
                    key = match.lower()
                    if key not in seen_lower:
                        seen_lower[key] = match

    # Combine original + new forms, remove duplicates
    final_list = list(set(candidates + list(seen_lower.values())))
    return final_list
final_candidate = extract_possessives(cleaned_text, final_candidate)

# GEOGRAPHIC
location_keywords = [
    "danau", "sungai", "gunung", "bukit", "hutan", "pantai", "laut", "teluk",
    "lembah", "pulau", "rawa", "gua", "kawah", "kota", "desa", "kampung",
    "kabupaten", "provinsi", "jalan", "alun-alun", "pelabuhan", "bandara",
    "terminal", "kerajaan", "istana", "candi", "benteng", "keraton", "pasar",
    "makam", "masjid", "gereja", "pura", "vihara", "klenteng"
]

def is_location_candidate(candidates):
    filtered_candidate = []
    for name in candidates:
        tokens = name.split()
        if not any(loc in token.lower() for token in tokens for loc in location_keywords):
            filtered_candidate.append(name)
    return filtered_candidate
final_candidate = is_location_candidate(final_candidate)


def clean_name(names):
    prefixes = ['si ', 'baginda ', 'sang ']
    cleaned_names = []

    for name in names:
        name_lower = name.lower()
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                name_lower = name_lower[len(prefix):]
                break  # Only remove one prefix
        cleaned_names.append(name_lower)

    return cleaned_names

final_candidate = clean_name(final_candidate)
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# FILTERING BASED ON AVAILABILITY
def clean_name_ngram_direct(text, character_list):
    tokens = [t.lower() for t in word_tokenize(text) if t.strip() != '']
    max_n = max(len(word_tokenize(candidate)) for candidate in character_list) if character_list else 1
    found_ngrams = {}
    
    for candidate in character_list:
        found_ngrams[candidate] = []
        for i in range(len(tokens) - max_n + 1):
            window = tokens[i:i+max_n]
            string_ngram = " ".join(window)
            if string_ngram.startswith(candidate):
                found_ngrams[candidate].append(string_ngram)

    # Track where each value appears
    value_to_keys = defaultdict(list)
    for key, values in found_ngrams.items():
        for value in values:
            value_to_keys[value].append(key)

    # Deduplicate
    for value, keys in value_to_keys.items():
        if len(keys) > 1:
            keys_sorted = sorted(keys, key=lambda k: (len(k.split()), len(k)), reverse=True)
            best_key = keys_sorted[0]
            for k in keys:
                if k != best_key and value in found_ngrams[k]:
                    found_ngrams[k].remove(value)

    matched_keys = [key for key, values in found_ngrams.items() if values]
    return matched_keys

final_candidate = clean_name_ngram_direct(cleaned_text, final_candidate)
final_candidate = sorted(final_candidate, key=len, reverse=True)
if final_candidate:
    progress.progress(40)
    status_text.text(f"Characters Extracted. Continue to cluster characters.")
    time.sleep(1)

# Clustering
import textdistance

# --- POINTERNYA BISA DISESUAIKAN JIKA PERLU
pointers = [
    'Ibu', 'Pak', 'Puteri', 'Permaisuri', 'Raja', 'Putera', 'Ayah', 'Istri',
    'Suami', 'Uwak', 'Menteri', 'Bunda', 'Anak', 'Kakak', 'Adik', 'Kakek',
    'Orang Tua', 'Tetangga', 'Putri', 'Beru Tandang', 'Putroe', 'Telangkai',
    'Tuhan', 'Abang'
]

# --- CLUSTERING FUNCTIONS ---
def compute_similarity(s1, s2):
    return {
        'jaro_winkler': textdistance.jaro_winkler(s1, s2),
        'jaro': textdistance.jaro(s1, s2),
        'ratcliff_obershelp': textdistance.ratcliff_obershelp(s1, s2)
    }

def cluster_without_pointers(characters_list, aliases_clusters, threshold):
    cluster_id = len(aliases_clusters) + 1
    for character in characters_list:
        found = False
        for key, cluster in aliases_clusters.items():
            for name in cluster:
                if character.endswith("nya") and name.endswith("nya"):
                    if compute_similarity(character[:-3].lower(), name[:-3].lower())['jaro_winkler'] >= threshold:
                        aliases_clusters[key].add(character)
                        found = True
                        break
                elif character.lower() in name.lower() or compute_similarity(character.lower(), name.lower())['jaro_winkler'] >= threshold:
                    aliases_clusters[key].add(character)
                    found = True
                    break
            if found:
                break
        if not found:
            aliases_clusters[f"person-{cluster_id}"] = {character}
            cluster_id += 1
    return aliases_clusters

def cluster_with_pointers(characters_with_pointer, pointers, threshold):
    all_pointer_item_clusters = []
    for p in pointers:
        pointer_cluster = {}
        pointer_cluster_id = 1
        one_token_list = []
        character_per_pointer = [
            character for character in characters_with_pointer
            if character.lower().startswith(p.lower())
        ]
        for character in character_per_pointer:
            character_token = character.split(" ")
            if len(character_token) == 1:
                one_token_list.append(character)
            else:
                found = False
                for key, cluster in pointer_cluster.items():
                    for name in cluster:
                        if character.endswith("nya") and character[:-3] in name:
                            cluster.append(character)
                            found = True
                            break
                        if character[len(p):].lower() in name[len(p):].lower() or compute_similarity(character[len(p):].lower(),name[len(p):].lower())['jaro_winkler'] >= threshold:
                            cluster.append(character)
                            found = True
                            break
                if not found:
                    pointer_cluster[pointer_cluster_id] = [character]
                    pointer_cluster_id += 1
        if one_token_list:
            if len(pointer_cluster) == 1:
                for item in one_token_list:
                    pointer_cluster[1].append(item)
            elif not pointer_cluster:
                pointer_cluster[pointer_cluster_id] = one_token_list
        all_pointer_item_clusters.extend(pointer_cluster.values())
    return all_pointer_item_clusters

def merge_clusters(aliases_clusters, pointer_clusters):
    final_clusters = {}
    counter = 1
    for cluster in pointer_clusters:
        final_clusters[f"Tokoh-{counter}"] = cluster
        counter += 1
    for key, value in aliases_clusters.items():
        final_clusters[f"Tokoh-{counter}"] = list(value)
        counter += 1
    return final_clusters

def cluster_character_aliases(characters_list, pointers):
    aliases_clusters = {}
    characters_with_pointer = [
        character for character in characters_list
        if any(character.lower().startswith(pointer.lower()) for pointer in pointers)
    ]

    characters_without_pointer = [character for character in characters_list if character not in characters_with_pointer]

    aliases_clusters = cluster_without_pointers(characters_without_pointer, aliases_clusters, 0.90)
    pointer_clusters = cluster_with_pointers(characters_with_pointer, pointers, 0.90)

    return merge_clusters(aliases_clusters, pointer_clusters)


clusters = cluster_character_aliases(final_candidate, pointers)
# Tampilkan sebagai DataFrame
df_clusters = pd.DataFrame([
    {"Tokoh": tokoh, "aliases": aliases}
    for tokoh, aliases in clusters.items()
])
if not df_clusters.empty:
    progress.progress(50)
    status_text.text(f"Characters Clustered. Continue to feature extraction.")
    time.sleep(1)


# CLASSIFICATION
import re

df_clusters["Sentences"] = [[] for _ in range(len(df_clusters))]

for idx, row in df_clusters.iterrows():
    aliases_lower = [alias.lower() for alias in row["aliases"]]
    matched_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(re.search(r'\b' + re.escape(alias) + r'\b', sentence_lower) for alias in aliases_lower):
            matched_sentences.append(sentence)

    df_clusters.at[idx, "Sentences"] = matched_sentences

# FEATURE EXTRACTION
df_exploded = df_clusters.explode("Sentences").reset_index(drop=True)

import pandas as pd

def is_dialogue(sentence):
    if isinstance(sentence, str):
        return int(any(q in sentence for q in ['"', '“', '”', '‘', '’']))
    return 0

# Apply to your DataFrame

df_exploded['is_dialogue'] = df_exploded['Sentences'].apply(is_dialogue)

import numpy as np

# Fungsi untuk membaca file lexicon
def load_inset(pos_path, neg_path):
    lexicon = {}
    for path in [neg_path, pos_path]:  # Tidak perlu pakai multiplier lagi
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()[1:]  # skip header
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0]
                        try:
                            score = float(parts[1])
                            lexicon[word] =  score
                        except ValueError:
                            continue  # skip if score is not a float
    return lexicon

# Load lexicon
lexicon = load_inset(pos_path, neg_path)

# Load pipeline
import stanza

# Load the Indonesian model once
nlp = stanza.Pipeline(lang='id', processors='tokenize,pos,lemma')

def verb_adj_sentiment_stanza(sentence, lexicon):
    doc = nlp(sentence)
    verbs = []
    adjs = []

    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower()
            if word.upos == 'VERB':
                verbs.append(lemma)
            elif word.upos == 'ADJ':
                adjs.append(lemma)

    relevant_words = verbs + adjs
    pos_count = sum(1 for word in relevant_words if lexicon.get(word, 0) > 0)
    neg_count = sum(1 for word in relevant_words if lexicon.get(word, 0) < 0)

    return pos_count - neg_count

# Terapkan ke semua baris dan kembalikan dua kolom baru
df_exploded['count_lexicon'] = df_exploded['Sentences'].apply(
    lambda x: verb_adj_sentiment_stanza(x, lexicon)
)

import stanza

# Load pipeline (jalankan sekali di awal)
nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', use_gpu=False)
is_subject_list = []

for idx, row in df_exploded.iterrows():
    sentence = row['Sentences']
    aliases = row['aliases']  # pastikan ini list, bukan string

    is_subject = 0
    try:
        doc = nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                for alias in aliases:
                    if alias.lower() in word.text.lower():
                        if word.deprel in ["nsubj", "nsubj:pass"]:
                            is_subject = 1
                            break  # optional: keluar dari alias loop
            if is_subject:
                break  # optional: keluar dari kalimat
    except Exception as e:
        print(f"Error processing sentence: {sentence}\n{e}")
    
    is_subject_list.append(is_subject)

# Tambahkan ke DataFrame
df_exploded['is_subject'] = is_subject_list
if not df_exploded.empty:
    progress.progress(80)
    status_text.text(f"Feature Extracted, continue to classification.")
    time.sleep(1)

# CLASSIFICATION
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
factory = StemmerFactory()
stemmer = factory.create_stemmer()
INDONESIAN_STOPWORDS = set(stopwords.words('indonesian'))

def preprocess_sentence(sentence):
    # Lowercase
    sentence = sentence.lower()
    
    # Tokenize using regex (words only, ignore numbers and special characters)
    tokens = re.findall(r'\b[a-zA-Z]+\b', sentence)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in INDONESIAN_STOPWORDS]
    
    # Stem each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

df_exploded["subject_sentences_stem"] = df_exploded['Sentences'].apply(preprocess_sentence)

import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load('/Users/ferroyudisthira/Desktop/Semesters/Final Project/3.main/model/best_model_one_svm_exp_tf.joblib')


# Ensure required columns exist
required_columns = ['is_dialogue', 'is_subject', 'count_lexicon', 'subject_sentences_stem']
group_columns = ['Tokoh', 'aliases']  # grouping keys

if not df_exploded.empty and all(col in df_exploded.columns for col in required_columns + group_columns):
    # Step 1: Predict per row
    predictions = loaded_model.predict(df_exploded[required_columns])
    df_exploded['predicted_label'] = predictions
    df_exploded['alias_str'] = df_exploded['aliases'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    # Step 2: Group by tokoh + alias and use majority vote
    grouped_df = (
        df_exploded
        .groupby(['Tokoh', 'alias_str'])['predicted_label']
        .agg(lambda x: Counter(x).most_common(1)[0][0])  # majority vote
        .reset_index()
    )

    if not grouped_df.empty:
        progress.progress(100)
        status_text.text(f"Classification Complete. Process Complete.")
        time.sleep(1)


# Preview results
# Display the text
if user_input:
    st.dataframe(grouped_df, use_container_width=True)
    