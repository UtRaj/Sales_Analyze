import nltk
import os
import re
import string
from collections import Counter

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade

import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)


if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, 'taggers/averaged_perceptron_tagger')):
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora/stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_dir)


model = SentenceTransformer('all-MiniLM-L6-v2')


index = faiss.IndexFlatL2(384)  # Dimension 384 for the 'all-MiniLM-L6-v2' model


def read_file(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    return content


def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def average_word_length(text):
    words = word_tokenize(text)
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(words)


def punctuation_density(text):
    total_chars = len(text)
    num_punctuation = sum([1 for char in text if char in string.punctuation])
    return num_punctuation / total_chars


def pos_density(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    total_words = len(tokens)
    pos_density = {tag: count / total_words for tag, count in pos_counts.items()}
    return pos_density


def sentence_complexity(text):
    sentences = sent_tokenize(text)
    complexity = sum([len(sent.split()) for sent in sentences]) / len(sentences)
    return complexity


def repetition_ratio(text):
    words = word_tokenize(text)
    unique_words = set(words)
    repetition_ratio = (len(words) - len(unique_words)) / len(words)
    return repetition_ratio


def dynamic_parameter_tracking(transcript_text):
    avg_word_len = average_word_length(transcript_text)
    punctuation_dens = punctuation_density(transcript_text)
    pos_dens = pos_density(transcript_text)
    sent_comp = sentence_complexity(transcript_text)
    rep_ratio = repetition_ratio(transcript_text)
    readability_score = flesch_kincaid_grade(transcript_text)

    updated_parameters = {
        'avg_word_len': avg_word_len,
        'punctuation_dens': punctuation_dens,
        'pos_dens': pos_dens,
        'sent_comp': sent_comp,
        'rep_ratio': rep_ratio,
        'readability_score': readability_score
    }

    return updated_parameters


def generate_score_and_justification(transcript_text, avg_word_len, punctuation_dens, pos_dens, sent_comp, rep_ratio, readability_score):
    prompt = f"""
    Analyze the following sales conversation transcript to determine the likelihood of the customer purchasing the course. Provide a score out of 100 for the likelihood of conversion. Also, justify the score with five bullet points, considering various aspects such as language quality, customer engagement, agent responsiveness, and any other relevant factors you identify.

    Transcript:
    {transcript_text}

    Additional Parameters:
    - Average Word Length: {avg_word_len}
    - Punctuation Density: {punctuation_dens}
    - Part-of-Speech Density: {pos_dens}
    - Sentence Complexity: {sent_comp}
    - Repetition Ratio: {rep_ratio}
    - Readability Score: {readability_score}

    After analyzing the text and parameters, provide a detailed score and justification:

    Use the provided parameters and transcript text to assess the likelihood of conversion. Consider factors such as the clarity and persuasiveness of language, the level of customer interest and engagement, the responsiveness and effectiveness of the agent, and any other relevant aspects that contribute to the likelihood of conversion.

    Conversion Score: _______/100
    Justification:
    - Bullet Point 1: Assess the language quality critically, considering any instances of jargon, unclear explanations, or overly salesy language.
    - Bullet Point 2: Evaluate customer engagement, highlighting any areas where the customer's interest waned or where the agent failed to address concerns adequately.
    - Bullet Point 3: Critique agent responsiveness and effectiveness, noting any instances of delayed responses, incomplete information, or lack of empathy.
    - Bullet Point 4: Identify potential obstacles to conversion, such as pricing concerns, uncertainty about course delivery, or customer objections that were not fully resolved.
    - Bullet Point 5: Consider the overall tone and atmosphere of the conversation, including any factors that may have positively or negatively influenced the customer's perception of the course and the agent's handling of the call.

    Additionally, analyze why the customer would be willing to buy the course and why they wouldn't during the conversation. Provide two bullet points for each scenario and justify which scenario is more likely to happen, based on the conversation analysis.

    Reasons Customer Would Buy the Course:
    - Bullet Point 1: Highlight the benefits and features of the course that align with the customer's needs and goals, emphasizing how it can help advance their career or skills.
    - Bullet Point 2: Address any concerns or objections raised by the customer, demonstrating how the course addresses those challenges effectively.

    Reasons Customer Wouldn't Buy the Course:
    - Bullet Point 1: Identify any unresolved concerns or objections raised by the customer that may prevent them from making a purchase decision.
    - Bullet Point 2: Consider any external factors or competing priorities mentioned by the customer that could impact their willingness or ability to enroll in the course.

    Justification for Likelihood of Conversion:
    - Provide a brief analysis comparing the reasons for buying and not buying the course based on the conversation. Justify which scenario is more likely to happen and why, considering the overall tone, customer engagement, and agent effectiveness during the conversation.

    Predictive Analysis:
    Based on the provided transcript and additional parameters, use your expertise to predict the likelihood of conversion. Consider factors such as the customer's level of interest, the agent's effectiveness in addressing concerns, and any potential obstacles to conversion.

    Salesperson Feedback:
    Lastly, provide feedback to the salesperson based on the transcript analysis. Highlight any mistakes made during the conversation and suggest improvements to enhance conversion rates. Justify your feedback with specific examples from the transcript.
    """

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_and_store_embeddings(text):
    embeddings = model.encode([text])
    index.add(embeddings)
    return embeddings

def main():
    st.title("Sales Conversation Analysis")

    # File upload
    uploaded_file = st.file_uploader("Upload sales conversation transcript", type=["txt"])
    if uploaded_file is not None:
        content = read_file(uploaded_file)
        cleaned_text = preprocess_text(content)
        updated_parameters = dynamic_parameter_tracking(cleaned_text)

        # Generate embeddings and store them
        embeddings = generate_and_store_embeddings(cleaned_text)

        # Generate score and justification
        score_and_justification = generate_score_and_justification(cleaned_text, **updated_parameters)

        # Display results
        st.markdown("**Score and Justification**")
        st.write(score_and_justification)

        st.markdown("**Additional Analysis**")

        # Reasons Customer Would Buy the Course
        st.markdown("**Reasons Customer Would Buy the Course**")
        st.markdown("- Highlight the benefits and features of the course that align with the customer's needs and goals, emphasizing how it can help advance their career or skills.")
        st.markdown("- Address any concerns or objections raised by the customer, demonstrating how the course addresses those challenges effectively.")

        # Reasons Customer Wouldn't Buy the Course
        st.markdown("**Reasons Customer Wouldn't Buy the Course**")
        st.markdown("- Identify any unresolved concerns or objections raised by the customer that may prevent them from making a purchase decision.")
        st.markdown("- Consider any external factors or competing priorities mentioned by the customer that could impact their willingness or ability to enroll in the course.")

        # Justification for Likelihood of Conversion
        st.markdown("**Justification for Likelihood of Conversion**")
        st.markdown("- Provide a brief analysis comparing the reasons for buying and not buying the course based on the conversation. Justify which scenario is more likely to happen and why, considering the overall tone, customer engagement, and agent effectiveness during the conversation.")

if __name__ == "__main__":
    main()
