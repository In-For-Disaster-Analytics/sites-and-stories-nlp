import ipywidgets as widgets #documentation: https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Events.html
import os
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index import  ServiceContext, set_global_service_context
from LLMs.utils import multi_copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

corral_llms = "/corral-tacc/projects/TDIS/LLMs/HF"
scratch = os.getenv('SCRATCH') 
system_prompt = """<|SYSTEM|># You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES]. 
Pay attention to the type of the properties, if you can't find data for a property set it to null. Don't make anything up and don't add any extra data. If you can't find any data for a node or relationship don't add it.
Only add nodes and relationships that are part of the schema. If you don't get any relationships in the schema only add nodes.

Example:
Schema: Nodes: [Person {age: integer, name: string}] Relationships: [Person, roommate, Person]
Alice is 25 years old and Bob is her roommate.
Nodes: [["Alice", "Person", {"age": 25, "name": "Alice}], ["Bob", "Person", {"name": "Bob"}]]
Relationships: [["Alice", "roommate", "Bob"]]"""



class LLM:
    """Large Language Model Wrapper Class
    """
    def __init__(self, embed_model = "local:BAAI/bge-small-en"): 
        #alternative embed: "local:all-MiniLM-L6-v2"
        """Initialize GUI for model. Allows for selecting local models through a drop down."""
        self.button = widgets.Button(
            description='Load Model',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Submit',
            icon='check'
        )
        self.embed_model = embed_model 
        self.model_loaded = False
        self.dropdown = widgets.Dropdown(
            options=['LLAMA2 7B',"LLAMA2 7B CHAT", 'LLAMA2 13B', 'LLAMA2 13B CHAT',"LLAMA2 70B CHAT",
                     'Zephyr'
                     ],
            value='LLAMA2 13B CHAT',
            description='Model:',
            disabled=False,
            )
        self.location = {"LLAMA2 7B":"Llama7b",
                        "LLAMA2 7B CHAT": "Llama7bchat",
                        "LLAMA2 13B":  "Llama-2-13b-hf",
                        "LLAMA2 13B CHAT": "Llama-2-13b-chat-hf",
                         "LLAMA2 70B CHAT":"Llama-2-70b-chat-hf",
                        "Zephyr": "zephyr-7b-beta"
                        }
        self.llm_select_widget = widgets.HBox([self.dropdown, self.button])
        
        self.button.on_click(self.on_button_clicked)

    def get_llm_path(self):
        """Looks up the path of the LLM based on the self.location Dictionary

        Returns:
            str: directory location for the LLM
        """
        return  os.path.join(corral_llms, self.location[self.dropdown.value])
        
    def on_button_clicked(self, path:str):
        """Copies then loads the LLM based on the dropdown widget selections. 

        Args:
            path (str): Directory Location
        """
        self.button.button_style = "warning"
        self.path = os.path.join(scratch,  self.location[self.dropdown.value])
        
        ## Multiprocessing copy of the LLM to ensure speedy transfers. 
        multi_copy( self.path, self.get_llm_path())
        self.load_llm()
        self.model_loaded=True
        self.button.button_style = 'success'


    def load_llm(self, system_prompt:str=system_prompt, context_window:int=4096, max_new_tokens:int=1024):
        """Loads the selected LLM. 

        Args:
            system_prompt (str, optional): Prompt used prior to the query to give the LLM directions. Defaults to system_prompt.
            context_window (int, optional): Total number of characters that can be loaded into the prompt at once. . Defaults to 4096.
            max_new_tokens (int, optional): Total number of characters the LLM can respond withl. Defaults to 1024.
        """
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
        self.model = AutoModelForCausalLM.from_pretrained(self.path, 
                                             device_map='auto', 
                                             torch_dtype=torch.float16, 
                                             rope_scaling={"type": "dynamic", "factor": 2},
                                             load_in_8bit=True
                                            ) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.llm = HuggingFaceLLM(
                context_window=context_window,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                generate_kwargs={"temperature": 0.1, "do_sample": True},
                # tokenizer_name=self.path,
                # model_name=self.path,
                model=self.model,
                tokenizer = self.tokenizer,
                device_map="balanced",
                model_kwargs={ "load_in_8bit": False, "cache_dir":f"{scratch}"},
            )
        self.set_service_context()

    def set_service_context(self):
        """Sets the model and embed model for querying
        """
        self.service_context = ServiceContext.from_defaults(
                num_output=256,  # The amount of token-space to leave in input for generation.

                llm=self.llm, embed_model=self.embed_model
            )
        
        set_global_service_context(self.service_context)