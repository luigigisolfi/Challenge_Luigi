import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

######################################
# Load the HuggingFaceHub API token from the .env file
######################################

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
print(HUGGINGFACEHUB_API_TOKEN)

######################################
# Load the LLM model from the HuggingFaceHub
######################################

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_new_tokens": 500}
)

######################################
# Create a PromptTemplate and LLMChain 
######################################
template = """Question: {question}

Answer: Write a python script to {question} using pyautocad. Don't forget to import the necessary python modules. Provide code only."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

######################################
# Run the LLMChain
######################################

question = "Draw a 3D cube with a 5 cm side"
response = llm_chain.run(question)
with open('output.py', 'w') as file:
    file.write(response)
    print(response)
with open('output.py', 'r+') as fp:
    # read and store all lines into list
    lines = fp.readlines()
    print(lines)

#discard everything that comes before "import"
pos = lines.index("import pyautocad\n")
with open('output.py', 'w') as f:
    f.writelines(lines[pos:])

    # start writing lines except the last line
    # lines[:-1] from line 0 to the second last line
    f.writelines(lines[:-1])
file.close()

######################################
#Run output.py script
######################################

import subprocess
subprocess.run(["python", "output.py"])