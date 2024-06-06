from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Creating the LLM Model
llm_model = Ollama(model = "llama3", temperature = 0)

# The input text
email_message = "Here's out itinerary for our upcoming trip to France.\n We leave from Atlanta, Georgia airport 9:45 pm, and arrive in Paris Charles de Gaulle Airport.\n\nSome sightseeing will follow for a couple of hours. We will then go shop for gifts to bring back home.\nThe next morning, at 7:35 am we'll drive to Belgium, Brussels. While in Brussels we want to explore the city to it's fullest."

# The target output format
# desired_format = {
#     "leave_time": "8:45 pm",
#     "leave_from": "Atlanta, Georgia",
#     "cities_to_visit": ["Paris", "Brussels"]
# }

# STEP 1: Defining Output Schemas
leave_time_schema = ResponseSchema(name="leave_time",
                                   description="When they are leaving. It's usually a numeric time of the day. If not available, write not found.")
leave_from_schema = ResponseSchema(name="leave_from",
                                   description="Where are they leaving from. It's usually a city, airport or state. If not available write not found.")
cities_to_visit_schema = ResponseSchema(name="cities_to_visit",
                                        description="Cities or towns they will be visiting on their trip. If not available write not found.")

# STEP 2: Creating a Response Schema by combining the Output Schemas
response_schema = [
    leave_time_schema,
    leave_from_schema,
    cities_to_visit_schema
]

# STEP 3: Setup the output parser & format instructions
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

# Preparing format instructions
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

# STEP 4: Our email template - updated with {format_instruction}
email_template = "From the following email, extract the following information:\nleave_time: when are they leaving from a vacations to France. If there's an actual time written, use it, if not write unknown.\nleave_from: where are they leaving from, the airport or city name and state if available.\ncities_to_visit: extract the cities they are going to visit. If there are more than one, put them in square brackets like '['cityone', 'citytwo']'.\nCreate the output as JSON with the following keys:\nleave_time\nleave_from\ncities_to_visit\nemail: {email}\n{format_instructions}"

# Creating the prompt template for llama3
prompt_template = ChatPromptTemplate.from_template(template=email_template)
# print(prompt_template)

# Inserting the Input Variable Value
request = prompt_template.format_messages(email=email_message, format_instructions=format_instructions)

# STEP 5: Sending the request to llama3 LLM for response
response = llm_model.invoke(request)
# print(response)
# print(type(response))

# STEP 6: Formatted Output Dictionary
output_dict = output_parser.parse(response)
print('FORMATTED OUTPUT', output_dict)
print('FORMATTED OUTPUT TYPE', type(output_dict))

print('Visited Cities', output_dict['cities_to_visit'])
cities = output_dict['cities_to_visit']

for city in cities:
    print('City:', city)