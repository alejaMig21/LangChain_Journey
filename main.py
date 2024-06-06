from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def travel_idea(interest, budget):
    # DEFINE SYSTEM Prompt, budget template --> PromptTemplate
    system_template = "You are a travel agent that helps with people trips about {interest} on a budget of {budget}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    # print(system_message_prompt)
    
    # DEFINE HUMAN Prompt --> PromptTemplate
    human_template = "Please give me an example itinerary"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # print(human_message_prompt)
    
    # Compile ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # print(chat_prompt)
    
    # Insert Variable --> ChatPromptTemplate
    request = chat_prompt.format_prompt(interest = interest, budget = budget).to_messages()
    # print(request)
    
    # Chat Request -- llama3
    chat = Ollama(model = "llama3")
    result = chat.invoke(request)
    return result

# CALLING THE TRAVEL FUNCTION
output = travel_idea('Hiking', '$1000')
print(output)