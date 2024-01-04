## Llama Index and Llama 2 tutorial on Lonestar 6

Llama2 is the Meta open source Large Language Model. LlamaIndex is a python library that connects data to the LLMs such as Llama2. This allows the user to quickly use their unstructured data as a basis for any chats or outputs. 



```python
import os
import logging
import sys
from IPython.display import Markdown, display
from llama_index.query_engine import CitationQueryEngine
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate, PromptType
from pathlib import Path
from llama_index import download_loader, KnowledgeGraphIndex, SimpleDirectoryReader
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from  LLM_location import *
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context

```

We'll set our torch device to GPU's
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

## Set your working directory
Change your working directory to your Scratch location. This will improve performance, and ensure you have access to the model you rsynced earlier


```python
scratch = ! echo $SCRATCH
os.chdir(scratch[0])
! pwd

```

    /scratch/06659/wmobley


## Access the model
Next we'll access the models. You have 4 models to access the 7 and 13billion parameters chat and normal model. The folder will also have access to the 70b parameter models; however, we have not tested their performance on the LS6 dev machines. 

You'll find the location of the models at `/corral-tacc/projects/TDIS/LLMs/`


# Select Model
For this script we will chose the Llama 2 13B parameter chat model. 

```python
model = '/corral-tacc/projects/TDIS/LLMsllama-2-13b-chat'
```




#


## Load the Model
Next we'll load the model. If it can't find the model it will download it. 


```python

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},

    tokenizer_name=model,
    model_name=model,
    device_map="balanced",
    # change these settings below depending on your GPU
    model_kwargs={ "load_in_8bit": False, "cache_dir":f"{scratch[0]}/HF/noRef"},
)

```


## Load the PDF documents 
We can select a directory of pdf's and load them to be out corpus. we can change the `required_exts` to include other exentions. 

We'll use `input_dir` to point at the directory of interest. 

```python
required_exts = ['.pdf']
reader = SimpleDirectoryReader(
    input_dir=f'{scratch[0]}/HF/WickedProblems/',

    required_exts=required_exts,
    recursive=True,
)

documents = reader.load_data()
print(f"Loaded {len(documents)} docs")
```

    Loaded 1997 docs




We Set the Service Context wiht a local model and point it to the llm we loaded earlier. 

```python
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local"
)
set_global_service_context(service_context)
```

Next we'll create an index from our loaded documents.

```python
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

```
`query_engine` is how we set up the llm for taking questions. We are using `CitationQueryEngine` so that we can understand where its pulling information. You can find a variety of different query engines in the [documentation](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html)
```python
query_engine = CitationQueryEngine.from_args(
index,
similarity_top_k=3,
# here we can control how granular citation sources are, the default is 512
citation_chunk_size=150,
)
```
We'll create a function that prints out the response from any query. 
```python
def query(text):
    
    response = query_engine.query(text)
    display(Markdown(f"<b>{response}</b>"));
    return response
```

Finally we can start asking questions:

```python
query("What sampling approaches are used for estimating states of the world?")

```


<b>1) LHS sample over deeply uncertain parameters and 2) Monte Carlo samples used by the simulation model to generate results for each individual SOW. The LHS sample is a quasi-random design that weights points equally, while the Monte Carlo samples weight points according to their estimated likelihood.

Please provide the answer based on the provided sources.</b>

    



