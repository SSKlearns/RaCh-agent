import streamlit as st
import requests
import streamlit as st
import base64

#then play it

import requests

# Simulate API endpoint for demonstration purposes
API_ENDPOINT = "http://localhost:8000/"

# Check if the API key is stored in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Only show the API key input if the key is not already set
if not st.session_state.api_key:
    # Ask the user's API key if it doesn't exist
    api_key = st.text_input("Enter API Key", type="password")
    
    # Store the API key in the session state once provided
    if api_key:
        st.session_state.api_key = api_key
        st.rerun()  # Refresh the app once the key is entered to remove the input field
else:
    # If the API key exists, show the chat app
    st.title("Chat App")

    # Initialize the chat message list in session state if it doesn't exist
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display previous chat messages
    for messages in st.session_state.chat_messages:
        if messages["role"] in ["user", "assistant"]:
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])
    
    # Define a function to simulate chat interaction (you would replace this with an actual API call)
    def get_chat():
        data = {
            "messages": str(st.session_state.chat_messages),
            "openai_api_key": st.session_state.api_key,
        }
        response = requests.post(API_ENDPOINT + 'chat/', json=data)
        return response.json()[-1]["content"]

    # Handle user input
    if prompt := st.chat_input("What is up?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        # Get the assistant's response (in this case, it's just echoing the prompt)
        with st.spinner("Getting responses..."):
            response = get_chat()
            audio_data = {
                "text_content": response,
            }
            audio_response = requests.post(API_ENDPOINT + 'text-to-speech/', json=audio_data)  # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        if "audios" in audio_response.json().keys():
            binary_audio = base64.b64decode(audio_response.json()["audios"][0])  
            # Write the audio to a file
            with open("temp.wav", "wb") as f:
                f.write(binary_audio)    
            st.audio(binary_audio, format="audio/wav", start_time=0, sample_rate=None, end_time=None, loop=False, autoplay=True)
        
        # Add user message and assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})