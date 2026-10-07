import streamlit as st
import requests
import time

st.set_page_config(page_title="Hybrid Retrieval Engine", layout="centered")

st.title("Hybrid Retrieval Engine")
st.markdown("###✈️ Flight 1073 Incident Investigation Database")
st.markdown("""
This platform demonstrates a production-grade **Retrieval-Augmented Generation (RAG) backend**. 
It combines Semantic Vector Search with Exact-Keyword BM25 Search, fused via Reciprocal Rank Fusion (RRF), and precision-ranked using a local AI Cross-Encoder.
""")

st.markdown("<small style='color: #94a3b8;'><i>Try searching for: black box telemetry, Mossad cyber intel, radar logs, or cockpit voice recorder</i></small>", unsafe_allow_html=True)
st.divider()

with st.form(key='search_form'):
    user_query = st.text_input("Search Database:", placeholder="e.g. black box telemetry...")
    submit_button = st.form_submit_button(label="Search")

if submit_button and user_query:
    with st.status("Executing Hybrid Search Pipeline...", expanded=True) as status:
        st.write("Initializing Vector & BM25 endpoints...")
        time.sleep(0.3) 
        st.write("Fusing candidates via Reciprocal Rank Fusion...")
        time.sleep(0.3)
        st.write("Running AI Cross-Encoder for precision scoring...")
        
        try:
            # 60s timeout prevents UI freezing if the backend model takes time to load weights
            response = requests.get(f"http://127.0.0.1:8000/hybrid-search?query={user_query}", timeout=60)
            
            if response.status_code == 200:
                results = response.json()
                status.update(label="Query Complete", state="complete", expanded=True)
                
                if results:
                    st.success(f"Retrieved {len(results)} highly relevant documents")
                    
                    for doc in results:
                        with st.container(border=True):
                            st.subheader(doc['title'])
                            st.write(doc['text'])
                            
                            # Futuristic confidence meter using a native progress bar with embedded text
                            score_val = min(doc['cross_encoder_score'], 1.0)
                            score_pct = score_val * 100
                            st.progress(score_val, text=f"Match Relevance: {score_pct:.1f}%")
                            
                else:
                    st.warning("No matching records found in the database.")
                    
            else:
                status.update(label="Backend Error", state="error", expanded=False)
                st.error(f"API Error: Received HTTP status code {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            status.update(label="Connection Failed", state="error", expanded=False)
            st.error("Cannot reach the backend. Ensure the FastAPI server is running on port 8000.")
        except requests.exceptions.Timeout:
            status.update(label="Timeout", state="error", expanded=False)
            st.error("The search request timed out.")
        except Exception as e:
            status.update(label="System Error", state="error", expanded=False)
            st.error(f"An unexpected error occurred: {str(e)}")