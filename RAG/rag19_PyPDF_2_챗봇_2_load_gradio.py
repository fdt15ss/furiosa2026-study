# 챗봇까지 맹그러봐요
# transformer 논문을 벡터 디비로 불러와서
# 요약, 인용 등등 할 수 있는 챗봇으로!!!

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
import gradio as gr

load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = "./_db/Faiss19"

start = time.time()
vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index19',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,

)
print("vector_store 시간:", time.time() - start)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

model = ChatOpenAI(
    model = 'gpt-5.6-luna',
    temperature=0,
    max_tokens = 1000,
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_template(
"""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
'주어진 정보로는 답변할 수 없습니다.라고 말씀해주세요.

컨텍스트 {context}
질문 : {input}
답변:
"""

)

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="영선봇!!")

# Gradio 실행
demo.launch()