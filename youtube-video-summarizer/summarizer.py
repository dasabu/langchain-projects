import os
import yt_dlp
import whisper
from typing import List, Dict, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import streamlit as st

class YoutubeVideoSummarizer:
    def __init__(self, llm: ChatOpenAI, embedding: OpenAIEmbeddings):
        # Initialize models
        self.llm = llm
        self.embedding = embedding
        # Initialize whisper
        self.whisper_model = whisper.load_model('base')
    
    def download_video(self, url: str) -> Tuple[str, str]:
        '''Download video and extract audio'''
        print(f'Downloading video from {url}...')
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'downloads/%(title)s.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = ydl.prepare_filename(info).replace('.webm', '.mp3')
                video_title = info.get('title', 'Unknown Title')
                print(f'Downloaded video successfully')
                return audio_path, video_title
        except Exception as e:
            print(f'Error downloading video: {e}')
            raise e
        
    def transcribe_audio(self, audio_path: str) -> str:
        '''Transcribe audio using Whisper'''
        print(f'Transcribing audio {audio_path}...')
        try:
            result = self.whisper_model.transcribe(audio_path)
            print(f'Transcribed audio successfully')
            return result['text']
        except Exception as e:
            print(f'Error transcribing audio: {e}')
            raise e
    
    def create_documents(self, text: str, video_title: str) -> List[Document]:
        '''Split text into chunks and create documents'''
        print(f'Creating documents for {video_title}...')
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            texts = text_splitter.split_text(text)
            documents = [
                Document(page_content=chunks, metadata={"source": video_title}) 
                for chunks in texts
            ]
            print(f'Created documents successfully')
            return documents
        except Exception as e:
            print(f'Error creating documents: {e}')
            raise e
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        '''Create vector store from documents'''
        print('Creating vector store...')
        # Create vector store
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=f'youtube_video_summary'
            )
            print(f'Created vector store successfully')
            return vector_store
        except Exception as e:
            print(f'Error creating vector store: {e}')
            raise e
    
    def generate_summary(self, documents: List[Document]) -> str:
        '''Generate summary using Langchain summarize chain'''
        print('Generating summary...')
        try:
            map_prompt = ChatPromptTemplate.from_template(
'''Write a concise summary of the following transcript section:
"{text}"

CONCISE SUMMARY:'''
            )
            combine_prompt = ChatPromptTemplate.from_template(
'''Write a detailed summary of the following video transcript sections:
"{text}"

Include:
- Main topics and key points
- Important details and examples
- Any conclusions or call to action

DETAILED SUMMARY:'''
            )
            summary_chain = load_summarize_chain(
                llm=self.llm,
                chain_type='map_reduce',
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=True
            )
            summary = summary_chain.invoke(documents)
            print(f'Generated summary successfully')
            return summary
        except Exception as e:
            print(f'Error generating summary: {e}')
            raise e
    
    def setup_qa_chain(self, vector_store: Chroma):
        '''Setup QA chain'''
        try:
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(),
                memory=memory,
                verbose=True
            )
            print(f'Setup QA chain successfully')
            return qa_chain
        except Exception as e:
            print(f'Error setting up QA chain: {e}')
            raise e

    def process_video(self, url: str) -> Dict:
        '''Process video and return summary and qa chain'''
        try:
            os.makedirs('downloads', exist_ok=True)
            
            # Download and process
            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            vector_store = self.create_vector_store(documents)
            summary = self.generate_summary(documents)
            print('\n\n\n\n>>> Summary: ', summary)
            qa_chain = self.setup_qa_chain(vector_store)
            
            # Clean up
            os.remove(audio_path)
            st.success(f'Processed video successfully')
            return {
                'summary': summary,
                'qa_chain': qa_chain,
                'title': video_title,
                'transcript': transcript
            }
        except Exception as e:
            st.error(f'Error processing video: {e}')
            raise e