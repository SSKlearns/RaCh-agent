# RaCh - A Rag based Chat application with Streamlit and FastAPI

This project is a quick PoC for a Retrieval-Augmented Generation (RAG) based chat application built with Streamlit for the frontend and FastAPI for the backend. It utilizes Chroma DB for storing embeddings, OpenAI's GPT for generating chat responses, and OpenAI's `text-embedding-3-large` model for creating embeddings. The application also leverages Langchain for enhanced functionality. Lastly, it uses Sarvam's `text-to-speech` API for converting responses into audio.

## Features

- Chat interface built with Streamlit
- FastAPI backend for handling requests
- Chroma DB for efficient storage and retrieval of embeddings
- OpenAI GPT for generating conversational responses
- OpenAI's `text-embedding-3-large` for creating text embeddings
- Integration with Langchain for advanced language processing
- Sarvam's `text-to-speech` API for audio responses

## Technologies Used

- [Streamlit](https://streamlit.io/) - For building an intuitive web application
- [FastAPI](https://fastapi.tiangolo.com/) - For creating the backend API
- [Chroma DB](https://www.trychroma.com/) - For storing and managing embeddings
- [OpenAI API](https://beta.openai.com/docs/) - For accessing GPT and embedding models
- [Langchain](https://langchain.com/) - For enhanced language processing capabilities
- [Sarvam](https://www.sarvam.ai/) - For converting responses into audio
- [Python](https://www.python.org/) - For everything because how could we survive without it

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SSKlearns/RaCh-agent.git
   cd RaCh-agent
  
2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:

    ```bash
    pip install -r requirements.txt

4. Usage

    Start the FastAPI backend:

    ```bash
    uvicorn app:app --host=localhost --reload
    ```
    
    In a new terminal, start the Streamlit frontend:
    
    ```bash
    streamlit run chatapp.py

  Open your web browser and go to `http://localhost:8501` to access the chat application.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.
