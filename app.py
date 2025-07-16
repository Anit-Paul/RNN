import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the LSTM model safely
try:
    model = load_model('lstm_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the tokenizer safely
try:
    with open('tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Function for predicting the next word
def predict_next_word(model, tokenizer, text, max_sequence_length):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) == 0:
            return None  # input text contains unknown words
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=1)[0]

        # Reverse lookup the predicted word
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return None
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Streamlit app UI
st.title("🔮 Next Word Prediction with LSTM")

input_text = st.text_input("Enter a sequence of words", "i am a")

if st.button("Predict Next Word"):
    max_seq_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_seq_len)

    if next_word is None:
        st.warning("Could not predict the next word. Try different input text.")
    elif next_word.startswith("Prediction error:"):
        st.error(next_word)
    else:
        st.success(f"Predicted next word: **{next_word}**")
