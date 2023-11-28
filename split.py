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
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
index_name = "./saved_index"
documents_folder = "./documents"


@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    if os.path.exists(index_name):
        print("index_name:"+index_name)
        print(type(index_name))
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./saved_index"),
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
    llm = OpenAI(model="gpt-4", temperature=0, max_tokens=1024)
    embed_model = OpenAIEmbedding()
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )
    #####################################################################################
    refine_template_str_on_enligh = ( """
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
    refine_template_str = ("""
        당신은 교수입니다.
        주어진 문맥에 따라 한국어로 하나 이상의 '개방형 질문'을 만들어야 합니다. 
        다음 기준을 반드시 준수해야 합니다."
        1. 질문은 주어진 문맥 내에서 명확하게 답변할 수 있는 내용이어야 합니다.
        2. 문맥상 답변할 수 없는 질문은 만들지 마세요.
        3. 각 질문에는 상세하고 근거에 입각한 답변이 필요합니다. 답변은 질문과 관련된 정확한 정보로 구성되어야 합니다."
        4. 제공된 출력 형식을 엄격하게 준수해야 합니다:
                           """)
    refine_template = PromptTemplate(refine_template_str)

    multiple_choice_template_str_on_enligh = ("""
    You are a Professor.
    Based on the given context, you need to create multiple-choice questions in Korean. 
    You MUST adhere to the following criteria:
    1. Each question should have four answer choices (1, 2, 3,4).
    2. The correct answer should be clear within the given context.
    3. Each question and answer should require a detailed and well-founded answer but only one is correct.
    4. You must strictly adhere to the provided output format:
    ----------------------
    Q : question
    A : 1. option1
        2. option2
        3. option3
        4. option4
    ----------------------
    """)
    multiple_choice_template_str = ("""
    귀하는 교수입니다.
    주어진 문맥에 따라 한국어로 객관식 문제를 작성해야 합니다. 
    다음 기준을 반드시 준수해야 합니다:
    1. 각 문제에는 4개의 답안(1, 2, 3, 4)이 있어야 합니다.
    2. 정답은 주어진 문맥 내에서 명확해야 합니다.
    3. 각 질문과 답에는 상세하고 근거가 있는 답변이 필요하지만 정답은 하나만 가능해야 합니다.
    4. 제공된 출력 형식을 엄격하게 준수해야 합니다:
    ----------------------
    Q : 질문
    A : 1. 옵션1
        2. 옵션2
        3. 옵션3
        4. 옵션4
    ----------------------
                                """)

    multiple_choice_template = PromptTemplate(multiple_choice_template_str)

    #####################################################################################
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
        ##############
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        multiple_choice_template = multiple_choice_template,
        refine_template = refine_template
        )

    response = query_engine.query(query_text)
    return str(response)

    

def get_nodes_from_document(document, splitter):
    nodes = splitter.split(document)
    return nodes


st.title("한국사 FAQ GENERATOR")
st.header("한국사 FAQ 챗봇에 오신걸 환영합니다")
st.write("나중에 추가설명 들어가면 됨")

st.sidebar.header("한국사 FAQ GENERATOR 설정")
api_key = st.sidebar.text_input("Enter your OpenAI API key here:", type="password")

index = None
index_name = "./saved_index"
documents_folder = "./documents"

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    index = initialize_index(index_name, documents_folder)

if index is None:
    st.sidebar.warning("Please enter your API key first.")

# 대화 내용 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 재실행 시 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# 응답
if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    # 사용자 메시지 히스토리에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = query_index(index, prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    # 봇 메시지 히스토리에 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
