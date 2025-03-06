import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Custom CSS for applying the new color palette
st.markdown("""
    <style>
        /* General styles */
        * {
            font-family: 'Poppins', sans-serif;
            font-size: 15px;
        }

        /* Background colors */
        .main {
            background-color: #FBF8EF;
        }
        /* Background section with combo color */
        .stTextArea textarea {
            background-color: #C9E6F0;
            border: 2px solid #78B3CE;
            border-radius: 10px;
            padding: 10px;
        }

        /* Headers */
        h1, h2 {
            color: #F96E2A;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* File upload section */
        .stFileUploader label {
            font-weight: bold;
            color: #F96E2A;
        }

        /* Result table */
        .stDataFrame {
            border-radius: 10px;
            background-color: #ffffff;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Score card block */
        .score-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #78B3CE;
            color: white;
            text-align: center;
            margin: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }

        /* Score card heading */
        .score-card h3 {
            font-size: 18px;
            margin-bottom: 10px;
        }

        /* Score card paragraph */
        .score-card p {
            font-size: 16px;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app title
st.title("AI Resume Screening & Candidate Ranking System 🚀")

# Job description input with header
st.header("Job Description 📝")
job_description = st.text_area("Enter the job description", placeholder="Write the job description here...")

# File uploader section
st.header("Upload Resumes 📄")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Processing resumes and ranking them
if uploaded_files and job_description:
    st.header("Ranking Resumes 📊")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores with score cards
    for idx, score in enumerate(scores):
        st.markdown(f"""
        <div class='score-card'>
            <h3>{uploaded_files[idx].name}</h3>
            <p>Score: {score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Display results in a sorted table
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)

    # Bar Chart: Visualizing Scores
    st.header("Resume Scores Bar Chart 📊")
    fig, ax = plt.subplots()
    ax.barh(results["Resume"], results["Score"], color="#78B3CE")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Resume", fontsize=12)
    ax.set_title("Resume Ranking based on Score", fontsize=14, color='#F96E2A')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#78B3CE')
    ax.spines['bottom'].set_color('#78B3CE')
    st.pyplot(fig)

    # Pie Chart: Visualizing Scores as Percentages
    st.header("Resume Scores Pie Chart 🍰")

    # Fresh technical color palette
    colors = ['#F96E2A', '#C9E6F0', '#78B3CE', '#FBF8EF']

    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(
        results["Score"],
        labels=results["Resume"],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops=dict(color="#2f3640"),
        wedgeprops={"edgecolor": "0.5", 'linewidth': 1, 'linestyle': 'solid'}
    )
    
    # Customizing the font size and style for the pie chart labels
    for text in texts:
        text.set_fontsize(12)
        text.set_color('#2f3640')  # Dark grey for better contrast
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_color('white')

    ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig2)








