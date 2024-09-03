import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import time
import shutil
import fitz
import re
import plotly.graph_objects as go
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv()   

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="Mahabharata GPT", layout="wide")
st.title("Mahabharata GPT")

# Function to clear Docs folder on page reload
@st.cache_resource
def clear_docs_folder():
    for folder in ["Docs", "Highlighted"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

# Clear Docs folder on page reload
clear_docs_folder()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = {}
if 'chunk_metadata' not in st.session_state:
    st.session_state.chunk_metadata = {}
if 'character_usage' not in st.session_state:
    st.session_state.character_usage = {"Ved Vyas": 0, "Yudhishthir": 0, "Arjun": 0, "Duryodhan": 0, "Karn": 0}

# Character profiles
character_profiles = {
    "Ved Vyas": "You are Ved Vyas, the author of the Mahabharata. You have a broad perspective on all events and characters. Your responses should be wise, impartial, and insightful.",
    "Yudhishthir": "You are Yudhishthir, known for your adherence to dharma and morality. Your responses should focus on ethical considerations and righteous conduct.",
    "Arjun": "You are Arjun, the skilled warrior and central character of the Bhagavad Gita. Your responses should emphasize duty, skill, and the warrior's code.",
    "Duryodhan": "You are Duryodhan, the primary antagonist of the Mahabharata. Your responses should reflect ambition, power dynamics, and a perspective that challenges traditional morality.",
    "Karn": "You are Karn, known for your loyalty and internal conflicts. Your responses should reflect on personal dilemmas, friendship, and the complexities of dharma."
}

# Custom prompt template to handle dynamic context switching based on personalities
prompt = ChatPromptTemplate.from_template(
    """
    You are {selected_personality} from the Mahabharata. {character_profile}

    Answer the following question based on the provided context and chat history. Your answer should reflect your character's perspective and personality.

    Your answer should be in Markdown format.
    Include the exact statement or verse where you found the answer and the page number or section, if applicable.
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {input}
    """
)

# Function to process and embed documents
def vector_embedding():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./Docs")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    # Store PDF paths and chunk metadata
    for doc in st.session_state.final_documents:
        source = doc.metadata['source']
        st.session_state.pdf_paths[source] = source
        if source not in st.session_state.chunk_metadata:
            st.session_state.chunk_metadata[source] = []
        st.session_state.chunk_metadata[source].append({
            'page': doc.metadata['page'],
            'content': doc.page_content
        })

# Function to find the exact location of a chunk in a PDF
def find_chunk_location(pdf_path, chunk_content, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Clean up the chunk content for better matching
    chunk_content = re.sub(r'\s+', ' ', chunk_content).strip()  
    
    # Search for the chunk content
    text_instances = page.search_for(chunk_content)
    
    if text_instances:
        return text_instances[0]
    
    # If exact match not found, try partial matching
    words = chunk_content.split()
    for i in range(len(words), 0, -1):
        partial_chunk = ' '.join(words[:i])
        text_instances = page.search_for(partial_chunk)
        if text_instances:
            return text_instances[0]
    
    return None

# Function to highlight text in PDF
def highlight_pdf(pdf_path, chunk_content, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    chunk_rect = find_chunk_location(pdf_path, chunk_content, page_num)
    if chunk_rect:
        highlight = page.add_highlight_annot(chunk_rect)
        highlight.update()
    
    output_path = f"./Highlighted/{os.path.basename(pdf_path)}"
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    return output_path

# Function to select appropriate character based on question
def select_character(question):
    question = question.lower()
    if any(word in question for word in ["dharma", "morality", "ethics", "righteousness"]):
        return "Yudhishthir"
    elif any(word in question for word in ["war", "battle", "skill", "duty"]):
        return "Arjun"
    elif any(word in question for word in ["power", "ambition", "rivalry", "politics"]):
        return "Duryodhan"
    elif any(word in question for word in ["dilemma", "friendship", "loyalty", "conflict"]):
        return "Karn"
    else:
        return "Ved Vyas"

# Function to generate word cloud
def generate_word_cloud(text):
    # Convert text into a DataFrame
    words = text.split()
    word_freq = pd.Series(words).value_counts()
    df = pd.DataFrame({'word': word_freq.index, 'freq': word_freq.values})
    
    # Generate word cloud with Plotly
    fig = px.treemap(df, path=['word'], values='freq',
                     color='freq', hover_data=['freq'],
                     color_continuous_scale='Blues')
    
    st.plotly_chart(fig, use_container_width=True)

# Function to plot character usage
def plot_character_usage():
    fig = go.Figure(data=[go.Bar(
        x=list(st.session_state.character_usage.keys()),
        y=list(st.session_state.character_usage.values())
    )])
    fig.update_layout(title="Character Usage", xaxis_title="Characters", yaxis_title="Number of Responses")
    st.plotly_chart(fig)

# Sidebar for file uploads, character selection, and analytics
with st.sidebar:
    st.header("Upload Mahabharata PDF Files")
    uploaded_files = st.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join("Docs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                vector_embedding()
            st.success("Vector Store DB Is Ready")
    
    st.header("Select Character")
    selected_character = st.selectbox("Choose a specific character (optional):", 
                                      ["Auto-select"] + list(character_profiles.keys()))

    st.header("Analytics")
    if st.button("Show Character Usage"):
        plot_character_usage()

    if st.button("Generate Word Cloud"):
        all_text = " ".join([q + " " + a for q, a, _ in st.session_state.chat_history])
        generate_word_cloud(all_text)

# Main chat interface
st.subheader("Chat with Mahabharata GPT")

# Display chat history
for i, (question, answer, character) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(f"**{character}:** {answer}")

# User input
user_question = st.chat_input("Ask a question about the Mahabharata:")

if user_question:
    if st.session_state.vectors is None:
        st.error("Please upload and process documents first.")
    else:
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            chat_history = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in st.session_state.chat_history])
            
            # Select character
            if selected_character == "Auto-select":
                selected_personality = select_character(user_question)
            else:
                selected_personality = selected_character

            # Update character usage
            st.session_state.character_usage[selected_personality] += 1

            start = time.process_time()
            try:
                response = retrieval_chain.invoke({
                    'input': user_question,
                    'chat_history': chat_history,
                    'selected_personality': selected_personality,
                    'character_profile': character_profiles[selected_personality]
                })
                
                process_time = time.process_time() - start

                # Display user question
                with st.chat_message("user"):
                    st.write(user_question)

                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(f"**{selected_personality}:** {response['answer']}")
                    st.caption(f"Response time: {process_time:.2f} seconds")

                # Add to chat history
                st.session_state.chat_history.append((user_question, response['answer'], selected_personality))

                # Show relevant document chunks and highlight PDFs
                with st.expander("Relevant Document Chunks"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"Chunk {i + 1}:")
                        st.write(doc.page_content)
                        
                        # Highlight the chunk in the original PDF
                        pdf_path = st.session_state.pdf_paths.get(doc.metadata['source'])
                        if pdf_path:
                            page_num = doc.metadata['page']
                            highlighted_pdf = highlight_pdf(pdf_path, doc.page_content, page_num)
                            with open(highlighted_pdf, "rb") as file:
                                st.download_button(
                                    label=f"Download Highlighted PDF for Chunk {i + 1}",
                                    data=file,
                                    file_name=f"highlighted_chunk_{i+1}.pdf",
                                    mime="application/pdf"
                                )
                        
                        st.write("---")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try rephrasing your question or check if the documents are processed correctly.")

# Export chat history
if st.button("Export Chat History"):
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["Question", "Answer", "Character"])
    csv = chat_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="mahabharata_chat_history.csv",
        mime="text/csv"
    )

# Clean up the Docs folder after processing
if st.sidebar.button("Clear uploaded documents"):
    shutil.rmtree("Docs")
    shutil.rmtree("Highlighted")
    os.makedirs("Docs", exist_ok=True)
    os.makedirs("Highlighted", exist_ok=True)
    st.session_state.vectors = None
    st.success("Documents cleared!")