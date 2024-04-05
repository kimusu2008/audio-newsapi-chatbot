import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
from langchain import OpenAI as oai
from langchain.chains import APIChain
import os
import base64

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ['NEWS_API_KEY'] = ""

docs = """API documentation:
Endpoint: https://newsapi.org
Top headlines /v2/top-headlines

This endpoint provides live top and breaking headlines for a country, specific category in a country, single source, or multiple sources. You can also search with keywords. Articles are sorted by the earliest date published first.

This endpoint is great for retrieving headlines for use with news tickers or similar.
Request parameters

    country | The 2-letter ISO 3166-1 code of the country you want to get headlines for. Possible options: ae ar at au be bg br ca ch cn co cu cz de eg fr gb gr hk hu id ie il in it jp kr lt lv ma mx my ng nl no nz ph pl pt ro rs ru sa se sg si sk th tr tw ua us ve za. Note: you can't mix this param with the sources param.
    category | The category you want to get headlines for. Possible options: business entertainment general health science sports technology. Note: you can't mix this param with the sources param.
    sources | A comma-seperated string of identifiers for the news sources or blogs you want headlines from. Use the /top-headlines/sources endpoint to locate these programmatically or look at the sources index. Note: you can't mix this param with the country or category params.
    q | Keywords or a phrase to search for.
    pageSize | int | The number of results to return per page (request). 20 is the default, 100 is the maximum.
    page | int | Use this to page through the results if the total results found is greater than the page size.

Response object
    status | string | If the request was successful or not. Options: ok, error. In the case of error a code and message property will be populated.
    totalResults | int | The total number of results available for your request.
    articles | array[article] | The results of the request.
    source | object | The identifier id and a display name name for the source this article came from.
    author | string | The author of the article
    title | string | The headline or title of the article.
    description | string | A description or snippet from the article.
    url | string | The direct URL to the article.
    urlToImage | string | The URL to a relevant image for the article.
    publishedAt | string | The date and time that the article was published, in UTC (+000)
    content | string | The unformatted content of the article, where available. This is truncated to 200 chars.

Use page size: 2
"""

from theme import (
    initPage,
)

initPage("Menara Work Order QnA Chatbot")

def transcribe_voice_to_text(audio_location):
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    audio_file= open(audio_location, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

def text_to_speech_ai(speech_file_path, api_response):
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    response = client.audio.speech.create(model="tts-1",voice="nova",input=api_response)
    response.stream_to_file(speech_file_path)
    
def render_suggestions():
    def set_query(query):
        st.session_state.suggestion = query

    suggestions = [
        "Current top tech news in Malaysia",
        "Top Facility Management News in the last 7 Days",
        "Facility Management Industry Trends 2023",
    ]
    columns = st.columns(len(suggestions))
    for i, column in enumerate(columns):
        with column:
            st.button(suggestions[i], on_click=set_query, args=[suggestions[i]])

def render_query():
    st.text_input(
        "Search",
        placeholder="Search, e.g. 'Top News in Malaysia'",
        key="user_query",
        label_visibility="collapsed",
    )

def callback():
    if st.session_state.my_recorder_output:
        
        audio_bytes = st.session_state.my_recorder_output['bytes']
        audio_location = "audio_file.wav"
        with open(audio_location, "wb") as f:
            f.write(audio_bytes)

        #Transcribe the saved file to text
        text = transcribe_voice_to_text(audio_location)
        st.session_state.audio = text

def get_query():
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "audio" not in st.session_state:
        st.session_state.audio = ""   
        
    user_query = st.session_state.suggestion or st.session_state.user_query or st.session_state.audio
    st.session_state.suggestion = None
    st.session_state.user_query = ""
    st.session_state.audio = ""
    
    return user_query

st.info(
    "Search news and summarize the results. Type a query to start or pick one of these suggestions or use the voice recorder to ask a question:"
)
st.markdown("<hr/>", unsafe_allow_html=True)

mic_recorder(key='my_recorder', 
                callback=callback,    
                start_prompt="Start asking your question",
                stop_prompt="Stop recording your question",
                use_container_width=True)

user_query = get_query()
render_suggestions()
render_query()
st.markdown("<hr/>", unsafe_allow_html=True)

if not user_query:
    st.stop()

MAX_ITEMS = 1

def chat_completion_call(text):
    llm = oai(temperature=0)
    chain = APIChain.from_llm_and_api_docs(llm, docs, headers={"X-Api-Key": os.environ["NEWS_API_KEY"]}, limit_to_domains=["https://newsapi.org"], verbose=True)
    response = chain.run(text)
    return response

container = st.container()
header = container.empty()
header.write(f"Looking for results for: _{user_query}_")
placeholders = []
for i in range(MAX_ITEMS):
    placeholder = container.empty()
    placeholder.status("Searching...")
    placeholders.append(placeholder)

items = chat_completion_call(user_query)

header.write(f"That's what I found about: _{user_query}_. **Summarizing results...**")
placeholders[i].info(f"{items}")

header.write("Search finished. Try something else!")

speech_file_path = 'audio_response.mp3'
text_to_speech_ai(speech_file_path, items)
st.audio(speech_file_path)

