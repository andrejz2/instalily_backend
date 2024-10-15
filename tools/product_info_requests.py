import dotenv
import os
import bs4
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import requests
from openai import OpenAI
import re

dotenv.load_dotenv()

api_key = os.getenv("OPENAI-API-KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = OpenAI()

# Model, part, Models-starting-with, 
# {brand}-Parts and {appliance}-Parts main has 'data-page-type'='Newfind'
# searchsuggestions/?term={term} has class='search-result__suggestions'
# nsearchresult/?ModelID={term} has class'search-result__nsearch'

# def load_model_info(soup):
#     if "starting with" in soup.find('h2').get_text():
#         return False
#     else:
#         return True



# def part_rec_common_symptoms_1(model_ID):
#     result = requests.get( "https://www.partselect.com/api/search/?searchterm=" + model_ID)
#     soup = bs4.BeautifulSoup(result.content, 'lxml')
#     if not confirm_if_valid_model(soup):
#         return "Invalid model ID"
#     common_symptoms = [re.sub(r'\n\nFixed by these parts\n+\s*Show All\n', '', elem.get_text()) for elem in soup.find_all('a', class_='symptoms')]
#     return common_symptoms

# # call this if user chooses one of the common symptoms
# def part_rec_common_symptoms_2(model_ID, symptom):
#     result = requests.get( "https://www.partselect.com/Models/"+model_ID+"/Symptoms/"+symptom+"/")
#     soup = bs4.BeautifulSoup(result.content, 'lxml')
#     recommended_parts_raw = soup.find_all('div', class_='symptoms align-items-center')
#     recommended_parts = []
#     for part in recommended_parts_raw:
#         title = part.find('a', class_='mb-sm-1 d-block bold').get_text()
#         part_number = part.find('div', class_='text-sm').find('a').get_text()
#         price = part.find('div', class_='mega-m__part__price').get_text(strip=True)
#         availability = part.find('div', class_='mega-m__part__avlbl').get_text(strip=True)
#         link = part.find('a', class_='mb-sm-1 d-block bold')['href']
#         recommended_parts.append({"title" : title, "part_number" : part_number, "price": price, "avalability": availability, "link": link})
#     return recommended_parts 

# end-user function
def determine_compatability(combined: str) -> str:
    part_ID, model_ID = combined.split('_+_')
    if confirm_if_valid_part(part_ID) == "Part number is invalid.":
        return "The provided part number is invalid."
    if confirm_if_valid_model(model_ID) == "Model number is invalid":
        return "The provided model number is invalid."
    
    if part_ID.startswith('PS'):
        part_ID = part_ID[2:]
    url = "https://www.partselect.com/api/Part/PartCompatibilityCheck?modelnumber="+model_ID+"&inventoryid="+part_ID+"&partdescription=undefined"
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    if "MODEL_PARTSKU_MATCH" in str(soup):
        return "The provided part and model are compatible."
    else:
        return "The provided part and model are not compatible."

# return webpage containing search results
def get_related_parts(combined: str) -> str:
    model_ID, search_term = combined.split('_+_')
    if confirm_if_valid_model(model_ID) == "Model number is invalid.":
        return "Model number is invalid."
    # category = llm_determine_part_category(search_term)
    related_parts_search_result = "https://www.partselect.com/Models/"+model_ID+"/Parts/?SearchTerm="+search_term
    return related_parts_search_result

# end-user function
# call this if the user does not choose one of the common symptoms
# def show_parts_by_section(model_ID):
#     parts_by_section = "https://www.partselect.com/Models/"+model_ID+"/#Sections"
#     return parts_by_section

# return webpage containing search results
# def get_related_parts_with_part_ID(model_ID: str, part_ID: str) -> str:
#     if confirm_if_valid_model(model_ID) == "Model number is invalid.":
#         return "Model number is invalid."
    
#     if confirm_if_valid_part(part_ID) == "Part number is invalid.":
#         return "Part number is invalid."
    
#     part_result = requests.get( "https://www.partselect.com/api/search/?searchterm=" + part_ID)
#     part_soup = bs4.BeautifulSoup(part_result.content, 'lxml')
#     title = part_soup.find('h1', class_='title-lg mt-1 mb-3').get_text().replace('-', '').strip()
#     category = llm_determine_part_category(title)
#     related_parts_search_result = "https://www.partselect.com/Models/"+model_ID+"/Parts/?SearchTerm=s"+category
#     return related_parts_search_result


def llm_extract_part_ID_from_query(query):
    # Call GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''Your job is to identify and extract a part ID and part numbers from a given query. 
             You work for a home appliance website. Sometimes, a query will contain multiple part 
             IDs. Only extract and respond with the first ID provided. Sometimes, a query will
             contain a part ID AND a model ID. Only extract and return the part ID. For example, 
             for the query "How much is part W11384469?", respond with just "W11384469". As another
             example, if a query is "Does part 8194001 fit with model 2213222N414?", only respond with
             "8194001".'''
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

def llm_extract_model_ID_from_query(query):
    # Call GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''Your job is to identify and extract a model ID and model numbers from a given query. 
             You work for a home appliance website. Sometimes, a query will contain multiple model 
             IDs. Only extract and respond with the first ID provided. Sometimes, a query will
             contain a model ID AND a part ID. Only extract and return the model ID. For example, 
             for the query "How do I fix my dishwasher model 2213222N414", respond with just "2213222N414".
             As another example, if a query is "Does part 8194001 fit with model 106106813067?", only respond with
             "106106813067".'''
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

# Helper function
def llm_determine_part_category(title):
    # Call GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             '''You are a title synthesizer for a home appliance website. You can identify the keyword and product category from a given title.
             Given a product title, respond only with the main category in one word. For example, for the title
             "LOWER RACK ROLLER WD12X26146", you should respond "Roller", or given "Utility Drawer Gasket - White WP2183037", you should respond "Gasket".'''
            },
            {
                "role": "user",
                "content": f"{title}"
            }
        ]
    )
    result = str(response.choices[0].message.content)
    return result

# Helper
def confirm_if_valid_model(model_ID):
    result = requests.get("https://www.partselect.com/api/search/?searchterm=" + model_ID)
    soup = bs4.BeautifulSoup(result.content, 'lxml')
    indicator = soup.find('div', role='main')
    result_type = indicator.has_attr('data-page-type')
    if result_type and indicator['data-page-type'] == 'MegaModel':
        if "Sections of the" in soup.find('h2').get_text():
            title = soup.find('h1', class_='title-main mt-3 mb-4').get_text()
            print(title)
            if "Refrigerator" in title:
                model_type = "Refrigerator"
            elif "Dishwasher" in title:
                model_type = "Dishwasher"
            else:
                return "Model number is invalid for dishwasher or refrigerator."
            return f"This is a valid model number for a {model_type}."
    return "Model number is invalid."

# Helper
def confirm_if_valid_part(part_ID):
    result = requests.get("https://www.partselect.com/api/search/?searchterm=" + part_ID)
    soup = bs4.BeautifulSoup(result.content, 'lxml')
    indicator = soup.find('div', role='main')
    result_type = indicator.has_attr('data-page-type')
    if result_type and indicator['data-page-type'] == 'PartDetail':
        return "This is a valid part number."
    return "Part number is invalid."

# def model_num_retreiver():
# review_chain = (
#     {"context": reviews_retriever, "question": RunnablePassthrough()}
#     | review_prompt_template
#     | chat_model
#     | StrOutputParser()
# )

# def repair_guide():
#     repair_guide_url = "https://www.partselect.com/Repair/"
#     return repair_guide_url