# streamlit_app.py (Streamlit Frontend)

import streamlit as st
import requests
import json

st.set_page_config(layout="wide")
st.header("CREDISHIELD: YOUR FRAUD DETECTION COPILOT")

# Initialize session state
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'follow_up_questions' not in st.session_state:
    st.session_state.follow_up_questions = []

col1, col2 = st.columns(2)

company_data = {
    "ABC Financials": "100001",
    "XYZ Pharmaceuticals": "100002",
    "PQR Automobiles": "100003",
    "LMN Technologies": "100004",
    "DEF Manufacturing": "100005"
}

with col1:
    speech_bool = st.button("TALK TO COPILOT")

query = ""

if speech_bool:
    st.write("Listening...")
    # Implement speech-to-text functionality here
    # For now, we'll use a text input as a placeholder
    query = st.text_input("Speech input (placeholder):", key="speech_input")
else:
    with col2:
        st.write("Company:", end=" ")
        company_name = st.selectbox("", options=list(company_data.keys()),label_visibility="collapsed")
        text_bool = st.button("CHAT WITH COPILOT")

        if text_bool:
            query = company_name

if query and not st.session_state.processing_complete:
    st.write(f"Your query is: {query}")
    
    with st.spinner("Processing..."):
        # Send request to Flask backend
        response = requests.post('http://localhost:5000/process', json={'query': query})
        if response.status_code == 200:
            st.session_state.initial_response = response.json()['response']
            st.session_state.processing_complete = True
            #st.session_state.company_data = json.loads(st.session_state.initial_response.split("Detailed Company Data:")[1].strip())
        else:
            st.error("Error processing query")

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("AI", st.session_state.initial_response))

# Display the chat history
st.write("Chat History:")
for role, message in st.session_state.chat_history:
    st.write(f"{role}: {message}")

# Continuous chat interface
with st.form(key='new_follow_up_form'):
    new_follow_up = st.text_input("Your question:", key="new_follow_up")
    submit_button = st.form_submit_button("Send")
    
    if submit_button and new_follow_up:
        with st.spinner("Processing your question..."):
            response = requests.post('http://localhost:5000/follow_up', json={'question': new_follow_up, 'company_data': st.session_state.company_data})
            if response.status_code == 200:
                follow_up_response = response.json()['response']
                st.session_state.chat_history.append(("User", new_follow_up))
                st.session_state.chat_history.append(("AI", follow_up_response))
                st.session_state.follow_up_questions.append(new_follow_up)
                st.experimental_rerun()

# Display all follow-up questions and their responses
for i, question in enumerate(st.session_state.follow_up_questions):
    st.write(f"User: {question}")
    st.write(f"AI: {st.session_state.chat_history[2*i+3][1]}")  # +3 because of initial query and response

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.initial_response = ""
    st.session_state.processing_complete = False
    st.session_state.company_data = None
    st.session_state.follow_up_questions = []
    st.experimental_rerun()
