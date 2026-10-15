# 23-2 카피

# pip install rank_bm25

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

######################### BM25 검색 #######################
# BM25는 키워드 기반 검색, Best Match 25
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# print("============== BM25 검색 결과 =================")
# for doc in bm25_docs:
#     print(doc.page_content)

#############################################################
#############################################################
########## 벡터 검색 ###########
import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import gradio as gr
from langchain_chroma import Chroma
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
 
load_dotenv() # vscode 자체에서

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    # model='text-embedding-3-large', # 3072
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = './_db/Chroma12/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma12",
    # documents=texts,
)

vector_retriever = vector_store.as_retriever(
    search_kwargs= {"k": 3}
)
vector_docs = vector_retriever.invoke(query)

# print("=================== 벡터디비 검색 결과 =======================")
# for doc in vector_docs:
#     print(doc.page_content)

#############################################################
#############################################################
########## 하이브리드 검색 ###########
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
        ],
        weights=[0.5, 0.5]
)

hybrid_docs = hybrid_retriever.invoke(query)

print("=================== 벡터디비 검색 결과 =======================")
for doc in hybrid_docs:
    print(doc.page_content)