import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory  # تغيير من langchain_core.memory إلى langchain.memory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv


load_dotenv()

# الحصول على API Key من المتغيرات البيئية
API_KEY = os.getenv('API_KEY')
# تعريف نموذج اللغة
llm = ChatOpenAI(
    model="gpt-4o-mini",  # تصحيح اسم النموذج (استخدم شرطات بدلاً من شرطات سفلية)
    api_key=API_KEY
)

# تعريف الذاكرة
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# تهيئة embeddings وقاعدة البيانات المتجهية
embeddings = OpenAIEmbeddings(api_key=API_KEY)
vector_db = Chroma(
    embedding_function=embeddings,
    collection_name="tutorial",
    persist_directory="./chroma_db"
)

# تهيئة استرجاع السياق
retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=vector_db.as_retriever()
)

# تهيئة قوالب الرسائل
system_template = """
استخدم المعلومات التالية للإجابة على سؤال المستخدم.

السياق:
{context}

تاريخ المحادثة:
{chat_history}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# تجميع قالب المحادثة
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# إنشاء سلسلة الاسترجاع التحادثية
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": chat_prompt}
)

# تعريف دالة الاستجابة للمحادثة
def chat_response(user_input):
    response = chain.invoke({"question": user_input})
    return response["answer"]