
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
import os

class DocumentInput(BaseModel):
    question: str = Field()

def load_and_setup_tools(directory_path, llm):
    tools = []
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    try:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if file.endswith((".pdf", ".docx", ".doc", ".txt", ".csv", ".xlsx", ".xls", ".pptx")):
                loaders = {
                    ".pdf": PyPDFLoader,
                    ".docx": Docx2txtLoader,
                    ".doc": Docx2txtLoader,
                    ".txt": TextLoader,
                    ".csv": CSVLoader,
                    ".xlsx": UnstructuredExcelLoader,
                    ".xls": UnstructuredExcelLoader,
                    ".pptx": UnstructuredPowerPointLoader
                }
                ext = os.path.splitext(file)[-1].lower()
                loader = loaders.get(ext)
                if loader:
                    pages = loader(file_path).load_and_split()
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    docs = text_splitter.split_documents(pages)
                    embeddings = OpenAIEmbeddings()
                    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
                    tool = Tool(
                        args_schema=DocumentInput,
                        name=file.split(".")[0],
                        description=f"useful when we want to answer questions about {file}",
                        func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                    )
                    tools.append(tool)
                    
    except Exception as e:
        print(f"Error loading documents: {e}")
        return ""
    return tools

def admin_agent(directory_path,llm):
    try:
        loaded_tools = load_and_setup_tools(directory_path)
        # llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")

        memory = ConversationBufferMemory(memory_key="chat_history",
                                        return_messages=True,
                                        output_key='output')

        agent = initialize_agent(agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                tools=loaded_tools,
                                llm=llm,
                                memory=memory,
                                verbose=False)
        return agent
    
    except Exception as e:
        return f"Error in admin_agent: {str(e)}"
    
# class DocumentInput(BaseModel):
#     question: str = Field()

# # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# def load_and_setup_tools(directory_path,llm):
#     tools = []

#     for file in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, file)
#         if file.endswith((".pdf", ".docx", ".doc", ".txt", ".csv", ".xlsx", ".xls", ".pptx")):
#             loaders = {
#                 ".pdf": PyPDFLoader,
#                 ".docx": Docx2txtLoader,
#                 ".doc": Docx2txtLoader,
#                 ".txt": TextLoader,
#                 ".csv": CSVLoader,
#                 ".xlsx": UnstructuredExcelLoader,
#                 ".xls": UnstructuredExcelLoader,
#                 ".pptx": UnstructuredPowerPointLoader
#             }
#             ext = os.path.splitext(file)[-1].lower()
#             loader = loaders.get(ext)
#             if loader:
#                 pages = loader(file_path).load_and_split()
#                 text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#                 docs = text_splitter.split_documents(pages)
#                 embeddings = OpenAIEmbeddings()
#                 retriever = FAISS.from_documents(docs, embeddings).as_retriever()
#                 tool = Tool(
#                     args_schema=DocumentInput,
#                     name=file.split(".")[0],
#                     description=f"useful when we want to answer questions about {file}",
#                     func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever))
#                 tools.append(tool)

#     return tools

# directory_path = "agreements"
# loaded_tools = load_and_setup_tools(directory_path)

# llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")

# memory = ConversationBufferMemory(memory_key="chat_history",
#                                   return_messages=True,
#                                   output_key='output')

# agent = initialize_agent(agent=AgentType.OPENAI_MULTI_FUNCTIONS,
#                          tools=loaded_tools,
#                          llm=llm,
#                          memory=memory,
#                          verbose=True)

# def chatbot_interface(query):
#     chat_history = []
#     if query == 'exit' or query == 'quit' or query == 'q' or query == 'f':
#         return 'Exiting'
#     if query == '':
#         return ''
#     #response = model({'question':query, 'chat_history':chat_history})
#     response = agent({"input":f"{query}"})
#     #chat_history.append(response['answer'])
#     return response['output']#['answer'], process_metadata(response['source_documents'])


# iface = gr.Interface(fn=chatbot_interface, 
#                      inputs="text", 
#                      outputs=["text"]#,"text"]
#                      )

# iface.launch()

