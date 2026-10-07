#12 - 4 카피
# Alt 쓰니까 느려짐, gpu로 돌려도 느려지고 cpu로 돌려도 느려짐
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
import time
from dotenv import load_dotenv
load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.
api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

#03. 임베딩

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    # model='text-embedding-3-large', # 3072
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = './_db/Chroma12/'
start = time.time()
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma12"
)
print("vector_store 시간:", time.time() - start)
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")
# 벡터 저장소에 저장된 문서 수 : 52

# query = " 삼성전자의 창업자는 누구인가요?"

# start = time.time()
# result = vector_store.similarity_search(query)  # k값 기본값은 4
# print("similarity_search 시간:", time.time() - start)
# print(f"검색 결과의 길이: {len(result)}")       # 4

################################### Retrievers ########################################
################################### 검색기 ###########################################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
# print(retriever)
# start = time.time()

# aaa = retriever.invoke(query)
# print("retriever.invoke 시간:", time.time() - start)

# print(f"검색된 관련 문서 수 : {len(aaa)}")
# print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")

print("=========================================================================")
################################### 모델 연결 #######################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature=0,
    max_tokens = 1000,
    api_key=api_key,
    base_url=base_url,
)
# response = model.invoke("삼성전자의 창업자는 누구인가요?")
# print("model의 답변 : ", response.content)
# print("=========================================================================")
# query_with_context = f"""
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# """

# # predict == invoke

# response = model.invoke(query_with_context)
# print("model의 응답 : ", response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
'주어진 정보로는 답변할 수 없습니다.라고 말씀해주세요.

컨텍스트 {context}
질문 : {input}
답변:
""")


# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)        # prompt | model 
rag_chain = create_retrieval_chain(retriever, docu_chain)       # 검색 | docu_chain
"""
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})

print(response)
print("===================== keys() ================================")
print(response.keys())
print("===================== context ================================")
print(response['context'][0].page_content)
print("===================== answer ===============================")
print(response['answer'])
"""
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="영선봇!!")

# Gradio 실행
demo.launch()