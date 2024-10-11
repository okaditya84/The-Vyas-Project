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
from deep_translator import GoogleTranslator
import time
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="Ved Vyas - Gita GPT", layout="wide")
st.title("Ved Vyas")

# Function to clear folders on page reload
@st.cache_resource
def clear_folders():
    folders = ["Docs", "Highlighted", "Translated", "VectorStore"]
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

# Clear folders on page reload
clear_folders()

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

# Custom prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the provided context and chat history, selecting the appropriate personality from Mahabharat (e.g., Arjun, Duryodhan, Yudhishtir, Karn) to answer based on the nature of the question.

    - If the question is about dharma or morality or leadership or nature of a being or being obedient, Yudhishtir should answer.
    - If it's about warfare, strategies, or valor, Arjun should respond.
    - For questions related to ambition, rivalry, or power, Duryodhan should answer.
    - For dilemmas or decisions or friendship or being kind, Karn should respond.

    Your answer should be in Markdown format.
    Include the exact statement or verse where you found the answer and the page number or section, if applicable.
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {input}
    
    Selected Personality: {selected_personality}
    """
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def translate_hindi_to_english(text, max_retries=3):
    translator = GoogleTranslator(source='hi', target='en')
    max_chunk_size = 1000  # Reduce chunk size for more manageable translation
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    translated_chunks = []

    for chunk in chunks:
        for attempt in range(max_retries):
            try:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to translate chunk after {max_retries} attempts: {str(e)}")
                    translated_chunks.append(chunk)  # Append original chunk if translation fails
                else:
                    time.sleep(1)  # Wait for 1 second before retrying
    
    return ' '.join(translated_chunks)

def vector_embedding():
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    all_documents = []
    
    for filename in os.listdir("Docs"):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join("Docs", filename)
            hindi_text = extract_text_from_pdf(pdf_path)
            
            # Save Hindi text
            with open(os.path.join("Docs", f"{filename[:-4]}_hindi.txt"), "w", encoding="utf-8") as f:
                f.write(hindi_text)
            
            # Translate and save English text
            try:
                english_text = translate_hindi_to_english(hindi_text)
                translated_path = os.path.join("Translated", f"{filename[:-4]}_english.txt")
                with open(translated_path, "w", encoding="utf-8") as f:
                    f.write(english_text)
                
                # Create document for vectorization
                doc = Document(page_content=english_text, metadata={"source": translated_path})
                all_documents.append(doc)
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
                continue
    
    if not all_documents:
        st.error("No documents were successfully processed. Please check your input files and try again.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)

    if not split_documents:
        st.error("No text chunks were created. Please check if the translated documents contain valid text.")
        return

    # Create vector store
    try:
        st.session_state.vectors = FAISS.from_documents(split_documents, st.session_state.embeddings)
        
        # Save vector store
        st.session_state.vectors.save_local("VectorStore")

        # Store metadata
        for doc in split_documents:
            source = doc.metadata['source']
            st.session_state.pdf_paths[source] = source
            if source not in st.session_state.chunk_metadata:
                st.session_state.chunk_metadata[source] = []
            st.session_state.chunk_metadata[source].append({
                'page': doc.metadata.get('page', 0),
                'content': doc.page_content
            })
        
        st.success("Vector store created successfully!")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

        
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

# Main chat interface
st.subheader("Chat with Gita GPT")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Gita PDF Files (Hindi)")
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

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message(f"user"):
        st.write(question)
    with st.chat_message(f"assistant"):
        st.write(answer)

# User input
user_question = st.chat_input("Ask a question about the Gita:")

if user_question:
    if st.session_state.vectors is None:
        st.error("Please upload and process documents first.")
    else:
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            chat_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
            
            # Dynamic personality selection based on the question
            if "dharma" in user_question.lower() or "morality" in user_question.lower():
                selected_personality = "Yudhishtir"
            elif "war" in user_question.lower() or "battle" in user_question.lower() or "strategy" in user_question.lower():
                selected_personality = "Arjun"
            elif "power" in user_question.lower() or "ambition" in user_question.lower() or "rivalry" in user_question.lower():
                selected_personality = "Duryodhan"
            else:
                selected_personality = "Karn"

            start = time.process_time()
            try:
                response = retrieval_chain.invoke({
                    'input': user_question,
                    'chat_history': chat_history,
                    'selected_personality': selected_personality
                })
                
                process_time = time.process_time() - start

                # Display user question
                with st.chat_message("user"):
                    st.write(user_question)

                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response['answer'])
                    st.caption(f"Response time: {process_time:.2f} seconds")

                # Add to chat history
                st.session_state.chat_history.append((user_question, response['answer']))

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

# Clean up folders after processing
if st.sidebar.button("Clear uploaded documents"):
    clear_folders()
    st.session_state.vectors = None
    st.success("Documents cleared!")