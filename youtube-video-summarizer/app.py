import os
from dotenv import load_dotenv
import streamlit as st
import torch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from summarizer import YoutubeVideoSummarizer

# Workaround for torch issue
torch.classes.__path__ = []

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Custom CSS for light gray background
st.markdown("""
    <style>
    .summary-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title('📺 YouTube Video Summarizer')
    st.write('Enter a YouTube URL to get a summary of the video')

    # Initialize models
    with st.spinner('Initializing models...'):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4o-mini')
        embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-small')
        summarizer = YoutubeVideoSummarizer(llm=llm, embedding=embedding)

    # Initialize session state for result and chat history
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Input for YouTube URL
    url = st.text_input('YouTube URL')
    submit = st.button('Submit')

    # Process video if URL is submitted and no result exists
    if submit and url and not st.session_state.result:
        with st.spinner('Processing video...'):
            try:
                st.session_state.result = summarizer.process_video(url)
            except Exception as e:
                st.error(f'Error processing video: {e}')
                return

    # Display video information if result exists
    if st.session_state.result:
        st.markdown(f'### 🔖 Title: {st.session_state.result["title"]}')
        
        st.markdown('### 📝 Summary')
        st.markdown(
            f'''<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            {st.session_state.result['summary']['output_text']}
            </div>''',
            unsafe_allow_html=True,
        )

        # Option to see full transcript
        with st.expander('📋 Full Transcript'):
            st.write(st.session_state.result['transcript'])

        # Option to see full documents
        with st.expander('📄 Full Documents'):
            for i, doc in enumerate(st.session_state.result['summary']['input_documents']):
                st.write(f'Document {i}: {doc.page_content}\n\n')

        # Interactive Q&A Section
        st.markdown('### ❓ Ask Questions About the Video')
        
        # Display chat history
        with st.container():
            for chat in st.session_state.chat_history:
                st.markdown(f"**You**: {chat['question']}")
                st.markdown(f"**Bot**: {chat['answer']}")
                st.markdown('---')

        # Input for user question
        with st.form(key='qa_form', clear_on_submit=True):
            user_question = st.text_input('Your Question:', key='question_input')
            submit_question = st.form_submit_button('Ask')

            if submit_question and user_question:
                try:
                    # Get response from qa_chain
                    with st.spinner('Thinking...'):
                        response = st.session_state.result['qa_chain']({
                            'question': user_question,
                            'chat_history': [(chat['question'], chat['answer']) for chat in st.session_state.chat_history]
                        })
                    # Append to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': response['answer']
                    })
                    # Rerun to update the display
                    st.rerun()
                except Exception as e:
                    st.error(f'Error processing question: {e}')

if __name__ == '__main__':
    main()