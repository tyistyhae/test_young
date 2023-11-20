import os
import streamlit as st
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
############
from llama_index import (
    ServiceContext,
    LLMPredictor,
    OpenAIEmbedding,
    PromptHelper,
)
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
###########
index_name = "./saved_index"
documents_folder = "./documents"


@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    if os.path.exists(index_name):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    #####################################
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
    embed_model = OpenAIEmbedding()
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
    )
    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query(query_text)
    #####################################
    # response = _index.as_query_engine().query(query_text)
    return str(response)

def get_nodes_from_document(document, splitter):
    nodes = splitter.split(document)
    return nodes

st.title("한국사 FAQ GENRATOR")
st.header("한국사 FAQ 챗봇에 오신걸 환영합니다")
st.write(
    "나중에 추가설명 들어가면 됨"
)

index = None
api_key = st.text_input("Enter your OpenAI API key here:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    index = initialize_index(index_name, documents_folder)


if index is None:
    st.warning("Please enter your api key first.")

text = st.text_input("Query text:", value="고구려에 대해 설명해줘")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            # f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}"
            "llm_col"
        )

    with embed_col:
        st.markdown(
            # f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
            "embed_col"
        )

    # if index.service_context.splitter:
    #     nodes = get_nodes_from_document(response, index.service_context.splitter)
    #     for node in nodes:
    #         prompt = f"Generate a question based on the content: {node}"
    #         question = index.service_context.llm_predictor.predict(prompt)
    #         answer = index.service_context.embed_model.predict(question)
    #         st.write("Node Content:")
    #         st.write(node)
    #         st.write("Generated Question:")
    #         st.write(question)
    #         st.write("Answer:")
    #         st.write(answer)
