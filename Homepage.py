import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image

st.set_page_config(
    page_title="Document Chat App",
    page_icon="📑",
)

st.title("Welcome to Document Questioning Application 👋🏼")

st.sidebar.success("Select a Page to Explore")
img1 = Image.open("Pipeline.jpg")
img2 = Image.open("Architecture.jpg")
with st.sidebar:
    add_vertical_space(5)
    st.write('Disclamer:')
    st.write('The application is still under development phase.')
    st.write('Further versions are soon to roll out with more functionality and better performance.')
    st.write('For any queries:-')
    st.write('Contact - 7044567565')
    
    st.write('Developed by Students of Computer Science and System Engineering')
    st.write('School of Computer Science Engineering, KIIT-DU')

st.markdown(
    """
    This is a simple document-based question-anwering application that is developed
    using **HuggingFace** and **Langchain** technology. The desiging is based on Python's
    **Streamlit** Library.

    ## Want to learn more about the technologies in use:
    - [**Hugging Face**](https://huggingface.co/)
    - [**LangChain**](https://python.langchain.com/)
    - [**Streamlit**](https://streamlit.io/)

    ### About❓
    The purpose of this application is to provide information reguarding the document
    provided by the user. It will read, analyse and answer questions that are relevent to
    the document in question. It can summarize  documents, extract key points from them,
    identify important entities or concepts mentioned therein, etc.
    The types of document that is supported by this application are:
    - PDF
    - Images

    The future models will be able to provide answers for other types of files too!

    ### Architecture🧱
    The Architecture of the application is based on the following components:
"""
)
st.image(
    img2,
    caption="Architecture Diagram of Document Chat Application",
    width= 800,
    channels="RGB"
)

st.markdown(
    """
    ### Pipeline 🔧
    The pipeline of the model along with all the components required to successfully run the application is
    visualized below

"""
)
st.image(
    img1,
    caption="Pipeline Diagram of Document Chat Application",
    width= 900,
    channels="RGB"
)