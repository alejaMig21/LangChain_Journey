from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import WikipediaLoader

def answer_question_about(person_name, question):
    # Use the Wikipedia Document Loader to help answer questions about someone, insert it as additional helpful context
    
    # STEP 1: Connect with llama3
    model = Ollama(model = "llama3", temperature = 0.5)
    
    # STEP 2: Load document
    loader = WikipediaLoader(query = person_name, load_max_docs = 1)
    context_text = loader.load()[0].page_content
    # print('Context Text', context_text)
    
    # STEP 3: HUMAN PROMPT - Format User Question
    human_template = "Answer this question:\n {question}\n Here is some extra context:\n {wiki_doc}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    # STEP 4: CHAT PROMPT & REQUEST - Compiling the main chat prompt & req
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    
    # REQUEST - Building the request to sent LLM
    request = chat_prompt.format_prompt(question = question, wiki_doc = context_text).to_messages()
    # print('REQUEST', request)
    
    # FINALLY AI RESPONSE - sending the request to Model
    response = model.invoke(request)
    
    print('AI RESPONSE:', response)

# Calling the method
answer_question_about('Brad Pitt', "Where was he born?")
answer_question_about('Stanley Kubrick', "What is the name of his first film?")
answer_question_about('Alan Turing', "When did he died?")