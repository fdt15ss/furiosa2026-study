# 맹그러봐

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

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
 
load_dotenv() # vscode 자체에서

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

embeddings = HuggingFaceEmbeddings(
    model_name='Qwen/Qwen3-Embedding-0.6B',
    model_kwargs={
        # "device" : "cuda", # Torch not compiled with CUDA enabled
        "device" : "cpu",
        "local_files_only" : True,
    }
)

DB_PATH = "./_db/Faiss20-2"

start = time.time()
vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index20-2',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,

)
print("vector_store 시간:", time.time() - start)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

model = ChatOpenAI(
    model = 'gpt-5.6-luna',
    # model = 'gpt-5.4-nano',
    temperature=0,
    max_tokens = 1000,
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_messages([
('system', """
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
'주어진 정보로는 답변할 수 없습니다.라고 말씀해주세요.

컨텍스트 {context}
"""
), MessagesPlaceholder(variable_name='chat_history'),
("human", '{input}')])

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

def answer_invoke(message, history):
    chat_history = []
    
    for item in history[-10:]: # 최근 10개의 메시지만 전달
        content = item["content"]

        if item['role'] == 'user':
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == 'assistant':
            chat_history.append(AIMessage(content=content))
    response = rag_chain.invoke({
        "input" : message,
        "chat_history" : chat_history
        })
    return response['answer']

# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="영선봇!!")

# Gradio 실행
demo.launch()

"""
1. 안녕
history = [
    {"role" : "user", "content": "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
]
2. 나는 임경빈이다.
history = [
    {"role" : "user", "content": "안녕"}
    {"role" : "assistant", "content" : "안녕하세요"}
    {"role" : "user", "content": "나는 임경빈이다."}
    {"role" : "assistant", "content" : "당신은 임경빈이군요"}
]
3. 내가 누구게.
hitstory = [
    {'role' : 'user', 'content' : '안녕'},
    {'role' : 'assistant', 'content' : '안녕하세요'},
    {'role' : 'user', 'content' : '나는 임경빈이다.'},
    {'role' : 'assistant', 'content' : '당신은 임경빈이군요'},
    {'role' : 'user', 'content' : '내가 누구게?'},
    {'role' : 'assistant', 'content' : '당신은 임경빈님입니다.'}
]
"""