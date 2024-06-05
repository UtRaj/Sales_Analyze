import nltk
import os

# Set NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Download NLTK data
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)



import streamlit as st
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tag import pos_tag
from textstat import flesch_kincaid_grade
import google.generativeai as genai
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Read the content of the provided file
def read_file(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    return content

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to calculate average word length
def average_word_length(text):
    words = word_tokenize(text)
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(words)

# Function to calculate punctuation density
def punctuation_density(text):
    total_chars = len(text)
    num_punctuation = sum([1 for char in text if char in string.punctuation])
    return num_punctuation / total_chars

# Function to calculate part-of-speech density
def pos_density(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    total_words = len(tokens)
    pos_density = {tag: count / total_words for tag, count in pos_counts.items()}
    return pos_density

# Function to calculate sentence complexity
def sentence_complexity(text):
    sentences = sent_tokenize(text)
    complexity = sum([len(TextBlob(sentence).parse().split('/')) for sentence in sentences]) / len(sentences)
    return complexity

# Function to calculate repetition ratio
def repetition_ratio(text):
    words = word_tokenize(text)
    unique_words = set(words)
    repetition_ratio = (len(words) - len(unique_words)) / len(words)
    return repetition_ratio


# Read the content of the provided file
def read_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    return content


# Function to generate a score and justification
def generate_score_and_justification(text, avg_word_len, punctuation_dens, pos_dens, sent_comp, rep_ratio, readability_score):
    prompt = f"""
    Analyze the following sales conversation transcript to determine the likelihood of the customer purchasing the course. Provide a score out of 100 for the likelihood of conversion. Also, justify the score with five bullet points, considering various aspects such as language quality, customer engagement, agent responsiveness, and any other relevant factors you identify.

    Transcript:
    {text}

    Additional Parameters:
    - Average Word Length: {avg_word_len}
    - Punctuation Density: {punctuation_dens}
    - Part-of-Speech Density: {pos_dens}
    - Sentence Complexity: {sent_comp}
    - Repetition Ratio: {rep_ratio}
    - Readability Score: {readability_score}

    After analyzing the text and parameters, provide a detailed score and justification:

    Use the provided parameters and transcript text to assess the likelihood of conversion. Consider factors such as the clarity and persuasiveness of language, the level of customer interest and engagement, the responsiveness and effectiveness of the agent, and any other relevant aspects that contribute to the likelihood of conversion.

    Score: _______/100
    Justification:
    - Bullet Point 1: Assess the language quality critically, considering any instances of jargon, unclear explanations, or overly salesy language.
    - Bullet Point 2: Evaluate customer engagement, highlighting any areas where the customer's interest waned or where the agent failed to address concerns adequately.
    - Bullet Point 3: Critique agent responsiveness and effectiveness, noting any instances of delayed responses, incomplete information, or lack of empathy.
    - Bullet Point 4: Identify potential obstacles to conversion, such as pricing concerns, uncertainty about course delivery, or customer objections that were not fully resolved.
    - Bullet Point 5: Consider the overall tone and atmosphere of the conversation, including any factors that may have positively or negatively influenced the customer's perception of the course and the agent's handling of the call.

    Additionally, analyze why the customer would be willing to buy the course and why they wouldn't during the conversation. Provide two bullet points

    for each scenario and justify which scenario is more likely to happen, based on the conversation analysis.

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

def main():
    st.title("Sales Conversation Analysis")

    # File upload
    uploaded_file = st.file_uploader("Upload sales conversation transcript", type=["txt"])
    if uploaded_file is not None:
        content = read_file(uploaded_file)
        cleaned_text = preprocess_text(content)
        avg_word_len = average_word_length(cleaned_text)
        punctuation_dens = punctuation_density(cleaned_text)
        pos_dens = pos_density(cleaned_text)
        sent_comp = sentence_complexity(cleaned_text)
        rep_ratio = repetition_ratio(cleaned_text)
        readability_score = flesch_kincaid_grade(cleaned_text)

        score_and_justification = generate_score_and_justification(cleaned_text, avg_word_len, punctuation_dens, pos_dens, sent_comp, rep_ratio, readability_score)

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
