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
#####################################################################################
from llama_index.prompts import PromptTemplate
#####################################################################################
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
###########################################################
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
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1024)
    embed_model = OpenAIEmbedding()
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )
    #####################################################################################
    refine_template_str = ( """
        you are a Profesor
        Based on the given context, you need to create one or more 'open-ended qusetions' in Korean. 
        You MUST obey tho following criteria :"
        1. The question must be something that can be answered clearly within the given context.
        2. DO NOT create a question that cannot be answered from the context
        3. Each question should require a detailed and well-founded answer. The answers should be composed of accurate information related to the question"
        4. You must strictly adhere to the provided output format : 
        ----------------------
        Q : question
        A : answer
    """)
    refine_template = PromptTemplate(refine_template_str)
    refine_template_str = ( """
    you are a Profesor
    Based on the given context, you need to create one or more 'open-ended qusetions' in Korean. 
    You MUST obey tho following criteria :"
    1. The question must be something that can be answered clearly within the given context.
    2. DO NOT create a question that cannot be answered from the context
    3. Each question should require a detailed and well-founded answer. The answers should be composed of accurate information related to the question"
    4. You must strictly adhere to the provided output format :
    ----------------------------------------------------------------------------------------------------------------------------------------------------                        
    Q: Which event or historical figure during the Joseon Dynasty is widely recognized for its significant impact on Korean history?
    1) Imjin War
    2) Sejong the Great
    3) Donghak Peasant Revolution
    4) Gwanghaegun's reign

    A: [Choose the correct option and provide a brief explanation of why you selected it.]

    Q: During the Japanese colonial period, what societal changes occurred in Korea, and how did these changes affect the Korean population?
    1) Industrialization and economic growth
    2) Preservation of traditional culture
    3) Forced labor and cultural suppression
    4) Political autonomy and independence

    A: [Choose the correct option and provide a concise explanation supporting your choice.]

    Q: The Korean War (6.25 War) had a profound impact on the historical trajectory of Korea. What key factors contributed to the outbreak of the Korean War?
    1) Ideological differences and political tensions
    2) Economic cooperation and regional stability
    3) International peace agreements
    4) Joint military exercises with neighboring countries

    A: [Choose the correct option and offer a detailed explanation of the factors leading to the outbreak of the Korean War.]

    Q: How did the scholar-official organization during the Joseon Dynasty shape the social structure of Korea?\n
    1) Fostering cultural diversity
    2) Promoting gender equality
    3) Influencing bureaucratic hierarchy
    4) Advocating for social upheaval

    A: [Choose the correct option and provide a comprehensive explanation of the impact of the scholar-official organization on Korean society.]

    Q: In the context of the Korean War, which geopolitical and historical factors can be identified as precursors to the conflict?
    1) Peaceful reunification efforts
    2) Cold War dynamics and division of Korea
    3) Cultural exchange programs
    4) Economic alliances with neighboring nations

    A: [Choose the correct option and elaborate on the geopolitical and historical factors that served as precursors to the Korean War.]
    """)
    refine_template = PromptTemplate(refine_template_str)
    #####################################################################################
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
        ##############
        # embed_model=llm,
        # query_wrapper_prompt=qa_template
        ##############
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        refine_template = refine_template
        )

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

text = st.text_input("Query text:", value="조선시대와 관련된 문제 3개와 그에 대한 정답을 알려줘")

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