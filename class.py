import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Efficient loading using caching
@st.cache_resource
def load_all():
    model = load_model('seq2seq_attention_model.h5', compile=False)
    tokenizer_eng = pickle.load(open('tokenizer_eng.pkl', 'rb'))
    tokenizer_urdu = pickle.load(open('tokenizer_urdu.pkl', 'rb'))
    config = pickle.load(open('config.pkl', 'rb'))
    return model, tokenizer_eng, tokenizer_urdu, config

model, tokenizer_eng, tokenizer_urdu, config = load_all()
max_encoder_len = config['max_encoder_len']
max_decoder_len = config['max_decoder_len']
reverse_urdu_index = {v: k for k, v in tokenizer_urdu.word_index.items()}
start_token = tokenizer_urdu.word_index['<start>']
end_token = tokenizer_urdu.word_index['<end>']

st.title("English to Urdu Translator")

# Beam width slider
beam_width = st.slider("Select Beam Width (higher = better translation, slower)", 1, 10, 3)

# Text input only (sample removed)
text = st.text_input("Enter English sentence:")

# Function to clean output
def clean_urdu_output(sentence):
    words = sentence.split()
    cleaned = [words[0]] if words else []
    for i in range(1, len(words)):
        if words[i] != words[i - 1]:
            cleaned.append(words[i])
    return ' '.join(cleaned)

# Warn about unknown words
if text.strip():
    unk_words = [w for w in text.lower().split() if w not in tokenizer_eng.word_index]
    if unk_words:
        st.warning(f"Unknown words (may reduce accuracy): {', '.join(unk_words)}")

# Beam search translation function
def beam_search_translate(input_text, beam_width=3):
    input_seq = tokenizer_eng.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_len, padding='post')

    beam = [([start_token], 0.0)]

    for i in range(1, max_decoder_len):
        all_candidates = []
        for seq, score in beam:
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            target_seq = np.zeros((1, max_decoder_len))
            for t, token in enumerate(seq):
                target_seq[0, t] = token

            output_tokens = model.predict([input_seq, target_seq], verbose=0)
            probs = output_tokens[0, i - 1, :]
            top_indices = np.argsort(probs)[-beam_width:]

            for idx in top_indices:
                prob = probs[idx]
                candidate_seq = seq + [idx]
                candidate_score = score + np.log(prob + 1e-10)
                all_candidates.append((candidate_seq, candidate_score))

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        beam = ordered[:beam_width]

        if all(seq[-1] == end_token for seq, _ in beam):
            break

    best_seq = beam[0][0]
    decoded_sentence = []
    for token_idx in best_seq:
        if token_idx in [start_token, end_token]:
            continue
        word = reverse_urdu_index.get(token_idx, '')
        if word:
            decoded_sentence.append(word)

    return ' '.join(decoded_sentence)

# Translate button
if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        result = beam_search_translate(text, beam_width=beam_width)
        result = clean_urdu_output(result)
        st.success(f"Urdu Translation: {result}")




