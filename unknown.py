import os
import glob
import numpy as np
import streamlit as st
import docx2txt
import faiss

# Import the modern Mistral SDK
from mistralai import Mistral

# ==============================================================================
# 1. CORE RAG PIPELINE FUNCTIONS
# ==============================================================================

def load_and_chunk_files(directory_path, chunk_size=1000, chunk_overlap=200):
    """Loads all .docx files from a folder and splits them into clean fragments."""
    chunks = []
    
    # Target only Word files (.docx)
    all_files = glob.glob(os.path.join(directory_path, "*.docx"))
        
    if not all_files:
        return []

    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            # Extract plain text directly from the Word file structural layers
            text = docx2txt.process(file_path)
            
            if not text or not text.strip():
                continue
                
            # Clean up excessive spacing and line endings
            text = " ".join(text.split())
            
            # Sliding window chunking algorithm
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                # Track metadata so the model knows which Word file the context came from
                chunks.append({"text": chunk, "source": filename})
                start += (chunk_size - chunk_overlap)
                
        except Exception as e:
            st.warning(f"⚠️ Failed to parse file {filename}: {str(e)}")
            
    return chunks


def build_vector_store(chunks, client, embedding_model="mistral-embed"):
    """Generates Mistral embeddings for all chunks and indexes them using FAISS."""
    if not chunks:
        return None, []
        
    texts_to_embed = [c["text"] for c in chunks]
    
    # Process text blocks in mini-batches safely
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        response = client.embeddings.create(
            model=embedding_model,
            inputs=batch
        )
        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)
        
    # Convert vectors to a float32 numpy array for FAISS index alignment
    dimension = len(all_embeddings[0])
    np_embeddings = np.array(all_embeddings).astype('float32')
    
    # Instantiate FlatL2 Vector index
    index = faiss.IndexFlatL2(dimension)
    index.add(np_embeddings)
    
    return index, chunks


def retrieve_context(query, index, chunks, client, top_k=4, embedding_model="mistral-embed"):
    """Embeds user queries and pulls the most contextually relevant document chunks."""
    if index is None or not chunks:
        return ""
        
    # Embed the user prompt
    response = client.embeddings.create(
        model=embedding_model,
        inputs=[query]
    )
    query_vector = np.array([response.data[0].embedding]).astype('float32')
    
    # Query vector neighborhood calculation
    distances, indices = index.search(query_vector, top_k)
    
    context_blocks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            chunk_data = chunks[idx]
            context_blocks.append(f"[Source: {chunk_data['source']}]\n{chunk_data['text']}")
            
    return "\n\n---\n\n".join(context_blocks)

# ==============================================================================
# 2. STREAMLIT UI & SESSION STATE MANAGEMENT
# ==============================================================================

st.set_page_config(page_title="HR policy Knowledge Base Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 HR policy Knowledge Base Chatbot")
st.write("Parse text layers from Word (.docx) formats, map embeddings, and run closed-domain QA.")

# Sidebar setup
with st.sidebar:
    st.header("🔑 Authentication & Setup")
    api_key = st.text_input("Mistral API Key", type="password", value=os.environ.get("MISTRAL_API_KEY", ""))
    
    st.header("📁 Word Source Directory")
    docs_dir = st.text_input("Folder path containing your 45 Word files", value="./my_word_documents")
    
    st.header("🧠 Model Tuning")
    llm_model = st.selectbox("LLM Engine", ["mistral-large-latest", "mistral-small-latest"], index=0)
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1, step=0.1)

# Initialize Session tracking
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
    st.session_state.document_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Action execution boundary
if st.sidebar.button("⚙️ Process & Index Word Files"):
    if not api_key:
        st.error("Please provide your Mistral API Key to calculate vector embeddings.")
    elif not os.path.exists(docs_dir):
        st.error(f"The path '{docs_dir}' does not exist locally.")
    else:
        with st.spinner("Parsing target .docx files and computing weights..."):
            client = Mistral(api_key=api_key)
            chunks = load_and_chunk_files(docs_dir)
            
            if not chunks:
                st.error("No valid .docx source files found in that directory. (Note: standard old .doc format is not supported).")
            else:
                index, cached_chunks = build_vector_store(chunks, client)
                st.session_state.vector_index = index
                st.session_state.document_chunks = cached_chunks
                st.sidebar.success(f"Successfully processed {len(cached_chunks)} text segments from Word files.")

# ==============================================================================
# 3. CHAT COMPONENT UTILITIES
# ==============================================================================

# Render interface memory trace
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt response cycle
if user_prompt := st.chat_input("Ask a question about your custom files..."):
    
    if not api_key:
        st.error("Please map your Mistral API Key via the left sidebar option.")
        st.stop()
        
    client = Mistral(api_key=api_key)
    
    # UI projection
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # 1. Pipeline Context matching
    extracted_context = ""
    if st.session_state.vector_index is not None:
        extracted_context = retrieve_context(
            user_prompt, 
            st.session_state.vector_index, 
            st.session_state.document_chunks, 
            client
        )
        
    # 2. Strict system instruction setup
    system_instruction = (
        "You are an expert internal AI database assistant. You have access to explicit proprietary knowledge text snippets below.\n"
        "Analyze the Context thoroughly. Answer the User Query objectively based *only* on the factual inputs. "
        "If the solution cannot be accurately deduced from the context, politely state that you do not possess the required facts.\n\n"
        f"--- CONTEXT START ---\n{extracted_context}\n--- CONTEXT END ---"
    )
    
    # 3. Streaming response generation
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        api_messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            stream_response = client.chat.stream(
                model=llm_model,
                messages=api_messages,
                temperature=temperature
            )
            
            for chunk in stream_response:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")
                    
            response_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An execution breakdown occurred: {str(e)}")