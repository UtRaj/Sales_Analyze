# Sales Analyze ( Sales Conversation Analysis )
This app is designed to analyze sales conversation transcripts to determine the likelihood of a customer purchasing a course. It utilizes natural language processing (NLP) techniques & various parameters and Google Gemini to generate a score and justification for the likelihood of conversion.

## Features

1. **Upload Transcript:** Users can upload a sales conversation transcript in TXT format.
   
2. **Dynamic Parameter Tracking**: The app dynamically tracks parameters such as average word length, punctuation density, part-of-speech density, sentence complexity, repetition ratio and readability score.

3. **Score and Justification Generation**: Based on the uploaded transcript and tracked parameters, the app generates a score out of 100 for the likelihood of conversion along with a detailed justification.

4. **Additional Analysis**: Provides further analysis on reasons why the customer would or wouldn't buy the course, along with justification for the likelihood of conversion.

5. **Predictive Analysis**: Allows users to make predictive assessments based on the conversation analysis.

6. **Salesperson Feedback**: Offers feedback to the salesperson based on the transcript analysis, highlighting mistakes made during the conversation and suggesting improvements to enhance conversion rates.


## **Installation**

1. Clone the repository:

```
git clone https://github.com/your-username/Sales.git
```

2. Navigate to the project directory:

```
cd Sales
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Run the Streamlit app:

```
streamlit run app.py
```




## **Screenshots**

Below are screenshots of the application:

### 1. Output Texts

![Opening Page](Screenshots/Output.png)

This is the output displaying predictive Analytics texts.


### 2. Output Graph

![Output Graph](Screenshots/Graphs.png)

This graph shows the results of the Predictive Analytics Texts depicting Grammatical data.




## **Usage**

1. Upload a sales conversation transcript in TXT format using the provided file uploader, you can use any file from "Testing Files" folder. 
   
2. Once the transcript is uploaded, the app will dynamically track parameters and generate a score and justification for the likelihood of conversion.
   
3. Additional analysis including reasons why the customer would or wouldn't buy the course, justification for the likelihood of conversion, predictive analysis, and salesperson feedback will be provided based on the transcript analysis.


## **Configuration**

Before running the application, ensure you have set up your Google API key in a .env file as follows:

```
GOOGLE_API_KEY=your-google-api-key
```



