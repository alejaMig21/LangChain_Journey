from langchain_community.llms import Ollama
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

llm = Ollama(model = "llama3", temperature = 0.25)

# STEP 1: SYSTEM PROMPT - Defining the AI's Job
system_template = "You are a helpful assistant that translate complex legal terms into plain and understandable terms."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# STEP 2: FEW SHOT TEMPLATES - Giving example output for AI
legal_text = "The provisions herein shall be severable, and if any provision or portion thereof si deemed invalid, illegal, or unenforceable by a court of competent jurisdiction, the remaining provisions or portions thereof shall remain in full force and effect to the maximun extent permitted by law."

example_input_one = HumanMessagePromptTemplate.from_template(legal_text)

plain_text = "The rules in this agreement can be separated. If a court decides that one rule or part of it is not valid, illegal, or cannot be enforced, the other rules will still apply and be enforced as much as they can under the law."

example_output_one = AIMessagePromptTemplate.from_template(plain_text)

# STEP 3: HUMAN TEMPLATE - The actual human prompt template
human_template = "{legal_text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# STEP 4: MAIN FEW SHOT TEMPLATE - Which includes examples
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_input_one, example_output_one, human_message_prompt])
# print(chat_prompt)

# STEP 5: USER INPUT - This is the complex text the user is giving to simplify by the AI
user_legal_text = "The grantor, being the fee simple owner of the real property herein described, conveys and warrants to the grantee, his heirs and assigns, all of the grantors's right, title, and interest in and to the said property, subject to all existing encumbrances, liens, and easements, as recorded in the official record of the county, and any applicable covenants, conditions, and restrictions affecting the property, in consideration of the sum of [purchase price] paid by the grantee."

# FINAL STEP: building the request object for AI & sending to LLM
request = chat_prompt.format_prompt(legal_text = user_legal_text).to_messages()

# REQUEST - Sending request to llama3
result = llm.invoke(request)
print(result)