# https://github.com/jsvine/pdfplumber

import pdfplumber
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import numpy as np
from scipy.spatial.distance import cosine
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
import chromadb
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import openai
import json
import ast

app = FastAPI()

# Define a Pydantic model for query requests
class QueryRequest(BaseModel):
    query: str
    openai_api_key: str
    
class ChatRequest(BaseModel):
    messages: str
    openai_api_key: str
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Extract a query on or related to the topic of  'sound'. Call this whenever the user asks a question about 'sound', or related, relevant topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query the user has about 'sound'. It can be anything from the user, such as 'What is sound?' or 'How does sound work?', or anything else demanding the knowledge of physics of sound.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_appreciation",
            "description": "Detect any form of appreciation by the user. Call this function if the user says anything positive, or sounds pleased with the results or conversation."
        }
    }
]

def read_document(file_path):
    x0 = 0    # Distance of left side of character from left side of page.
    x1 = 0.5  # Distance of right side of character from left side of page.
    y0 = 0  # Distance of bottom of character from bottom of page.
    y1 = 1  # Distance of top of character from bottom of page.

    all_content = ""
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            width = page.width
            height = page.height

            # Crop pages
            left_bbox = (x0*float(width), y0*float(height), x1*float(width), y1*float(height))
            page_crop = page.crop(bbox=left_bbox)
            left_text = page_crop.extract_text()

            left_bbox = (0.5*float(width), y0*float(height), 1*float(width), y1*float(height))
            page_crop = page.crop(bbox=left_bbox)
            right_text = page_crop.extract_text()
            page_context = ' '.join([left_text, right_text])
            all_content += page_context.replace('\n', ' ').replace('Rationalise', '')
            
        # close the pdf file
        pdf.close()
        
    return all_content

def sentence_chunker(text):
    # Split the text into sentences
    sentences = text.replace("Fig.", "Fig").split('. ')
    return sentences

def get_embedding(sentences, openai_api_key, model="text-embedding-3-large", dimensions=1024):
    # Get the embedding of a single sentence
    embeddings = OpenAIEmbeddings(
        model = model,
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        dimensions = dimensions,
        openai_api_key = openai_api_key
    )
    sentence_embeddings = embeddings.embed_documents(sentences if isinstance(sentences, list) else [sentences])
    return sentence_embeddings

def sentence_merger(sentences, embeddings, similarity_threshold=0.8, index_difference_threshold=1):
    grouped_sentences = []
    current_group = [sentences[0]]
    current_group_indices = [0]  # Track indices of sentences in the current group
    
    for i in range(1, len(sentences)):
        # Calculate cosine similarity
        sim = 1 - cosine(embeddings[i-1], embeddings[i])
        
        # Check if the current sentence is similar and close in index to the previous one
        if sim >= similarity_threshold and (i - current_group_indices[-1]) <= index_difference_threshold:
            current_group.append(sentences[i])
            current_group_indices.append(i)
        else:
            grouped_sentences.append(current_group)
            current_group = [sentences[i]]
            current_group_indices = [i]
    
    # Add the last group if it's not empty
    if current_group:
        grouped_sentences.append(current_group)
    
    return [" ".join(group) for group in grouped_sentences]

def vector_db(openai_api_key, sentences):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        dimensions=1024,
        openai_api_key=openai_api_key
    )

    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("embeddings_store")
    vector_store = Chroma(
        client=persistent_client,
        collection_name="embeddings_store",
        embedding_function=embeddings,
    )

    documents = [
        Document(
            page_content=sentence,
            metadata={"source": "pdf"},
            id=str(uuid4()),
        ) for sentence in sentences
    ]

    vector_store.add_documents(documents=documents)
    return

async def query_db(openai_api_key, query):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,
        openai_api_key=openai_api_key
    )
    
    # Embed the user query
    query_embedding = embeddings.embed_documents([query])[0]
    try:
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_collection("embeddings_store")
        vector_store = Chroma(
            client=persistent_client,
            collection_name="embeddings_store",
            embedding_function=embeddings,
        )
    except Exception as e:
        print("uploading pdf")
        await upload_pdf("iesc111.pdf", openai_api_key)
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_collection("embeddings_store")
        vector_store = Chroma(
            client=persistent_client,
            collection_name="embeddings_store",
            embedding_function=embeddings,
        )
        
    # Search for top 5 most similar documents to the query
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        query_embedding,
        k=5,
    )
    for i in range(len(results)):
        if results[i][1] > 1.2:
            results = results[:i]
            break
    # Extract the top relevant context from the results
    context = "\n\n".join([result.page_content for result, _ in results])
    print(context)
    return context

def generate_response_with_gpt4(query, openai_api_key, model="gpt-4o-mini"):
    # Retrieve the most relevant context from the vector database
    context = query_db(openai_api_key, query)
    
    if not context:
        return "Sorry, I couldn't find any relevant information in the PDF."

    # Define the prompt template
    template = """
    You are a helpful assistant. The following is some context based on a document:

    {context}

    Based only on the context above, answer the following question:
    {query}
    """

    # Set up the PromptTemplate with the context and user query
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )

    # Create the final prompt by formatting with the actual context and query
    final_prompt = prompt.format(context=context, query=query)
    
    messages = [
        (
            "system",
            "You are only going to respond based to the queries based on the given context.",
        ),
        ("human", final_prompt),
    ]

    # Initialize OpenAI's GPT-4
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_api_key,
    )
    
    # Pass the prompt to GPT-4 to generate a response
    response = llm.invoke(messages)

    return response

async def detect_appreciation(openai_api_key):
    return "Appreciation detected! Please frame a response that says thank you kindly. Give all credits to Sarvesh Shashikumar, for he is the one who made this possible. Ask them if they would like to hire him for an intern role and give his personal details. Sarvesh's number: 9930932102, Sarvesh's email: sarveshshashikumar0908@gmail.com"

@app.post("/upload-pdf/")
async def upload_pdf(file, openai_api_key: str):
    if type(file) == str:
        file = file
    else:
        file = file.file
    content = read_document(file)
    sentences = sentence_chunker(content)
    embeddings = get_embedding(sentences, openai_api_key)
    merged_sentences = sentence_merger(sentences, embeddings)
    vector_db(openai_api_key, merged_sentences)
    return {"message": "PDF content has been processed and stored in the vector database."}

@app.post("/query/")
async def query_db_endpoint(request: QueryRequest):
    response = generate_response_with_gpt4(request.query, request.openai_api_key)
    return {"response": response}

@app.post('/chat/')
async def chat_endpoint(request: ChatRequest):
    request.messages = ast.literal_eval(request.messages)
    llm = openai.OpenAI(api_key=request.openai_api_key)
    
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=request.messages,
        tools=tools,
        temperature=0
    )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(tool_call)
        arguments = json.loads(tool_call.function.arguments)
        function_name = tool_call.function.name

        # Get the corresponding function using `globals()` if it's defined in the global scope
        # Replace this with `locals()` or another mechanism if it's in a different scope
        function_to_call = globals().get(function_name)
        if function_to_call:
            # Call the function with the extracted arguments
            result = await function_to_call(openai_api_key=request.openai_api_key, **arguments)  # Pass the arguments as keyword arguments
            # Create a message containing the result of the function call
            function_call_result_message = {
                "role": "tool",
                "content": json.dumps({
                    "query": arguments,
                    "query_context": result
                }),
                "tool_call_id": response.choices[0].message.tool_calls[0].id
            }
            
            completion_payload = {
                "model": "gpt-4o-mini",
                "messages": request.messages + [response.choices[0].message] + [function_call_result_message],
            }
            response = llm.chat.completions.create(
                model=completion_payload["model"],
                messages=completion_payload["messages"],
            )
            
            request.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return request.messages
        
        request.messages.append({"role": "assistant", "content": "Sorry, I couldn't find any relevant information."})
        
        return request.messages
    
    print('No tool call')
    request.messages.append({"role": "assistant", "content": response.choices[0].message.content})
    return request.messages
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)