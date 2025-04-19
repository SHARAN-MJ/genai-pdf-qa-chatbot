#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
from langchain.document_loaders import PyPDFLoader

# File name
file_path = "tech.pdf"

# Confirm file existence and load
if os.path.isfile(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print("PDF loaded successfully.")
    print(pages[0].page_content if pages else "PDF is empty.")


# In[6]:


from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


# In[7]:


from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-4', temperature=0)


# In[8]:


from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


# In[9]:


from langchain.chains import RetrievalQA
question = "what is the definition of technology"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


# In[10]:


result = qa_chain({"query": question})
print("Question: ", question)
print("Answer: ", result["result"])


# In[ ]:





# In[ ]:





# In[ ]:




