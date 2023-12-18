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

```


```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[14], line 1
    ----> 1 import torch
          2 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ModuleNotFoundError: No module named 'torch'


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




```python
model = llm_location()

```


    Dropdown(description='Number:', index=3, options=('LLAMA2 7B', 'LLAMA2 7B CHAT', 'LLAMA2 13B', 'LLAMA2 13B CHA…


## Select Model
For this script we will chose the Llama 2 13B parameter chat model. 


```python
display(model.dropdown)
```

## Load the Model
Next we'll load the model. If it can't find the model it will download it. 


```python

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
#     query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model.get_llm_path(),
    model_name=model.get_llm_path(),
    device_map="balanced",
    # change these settings below depending on your GPU
    model_kwargs={ "load_in_8bit": False, "cache_dir":f"{scratch[0]}/HF/noRef"},
)#"torch_dtype": torch.float16,
print(selected_model)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    File ~/.local/lib/python3.11/site-packages/llama_index/llms/huggingface.py:156, in HuggingFaceLLM.__init__(self, context_window, max_new_tokens, system_prompt, query_wrapper_prompt, tokenizer_name, model_name, model, tokenizer, device_map, stopping_ids, tokenizer_kwargs, tokenizer_outputs_to_remove, model_kwargs, generate_kwargs, is_chat_model, messages_to_prompt, callback_manager)
        155 try:
    --> 156     import torch
        157     from transformers import (
        158         AutoModelForCausalLM,
        159         AutoTokenizer,
        160         StoppingCriteria,
        161         StoppingCriteriaList,
        162     )


    ModuleNotFoundError: No module named 'torch'

    
    The above exception was the direct cause of the following exception:


    ImportError                               Traceback (most recent call last)

    Cell In[13], line 1
    ----> 1 llm = HuggingFaceLLM(
          2     context_window=4096,
          3     max_new_tokens=2048,
          4     generate_kwargs={"temperature": 0.0, "do_sample": False},
          5 #     query_wrapper_prompt=query_wrapper_prompt,
          6     tokenizer_name=model.get_llm_path(),
          7     model_name=model.get_llm_path(),
          8     device_map="balanced",
          9     # change these settings below depending on your GPU
         10     model_kwargs={ "load_in_8bit": False, "cache_dir":f"{scratch[0]}/HF/noRef"},
         11 )#"torch_dtype": torch.float16,
         12 print(selected_model)


    File ~/.local/lib/python3.11/site-packages/llama_index/llms/huggingface.py:164, in HuggingFaceLLM.__init__(self, context_window, max_new_tokens, system_prompt, query_wrapper_prompt, tokenizer_name, model_name, model, tokenizer, device_map, stopping_ids, tokenizer_kwargs, tokenizer_outputs_to_remove, model_kwargs, generate_kwargs, is_chat_model, messages_to_prompt, callback_manager)
        157     from transformers import (
        158         AutoModelForCausalLM,
        159         AutoTokenizer,
        160         StoppingCriteria,
        161         StoppingCriteriaList,
        162     )
        163 except ImportError as exc:
    --> 164     raise ImportError(
        165         f"{type(self).__name__} requires torch and transformers packages.\n"
        166         "Please install both with `pip install transformers[torch]`."
        167     ) from exc
        169 model_kwargs = model_kwargs or {}
        170 self._model = model or AutoModelForCausalLM.from_pretrained(
        171     model_name, device_map=device_map, **model_kwargs
        172 )


    ImportError: HuggingFaceLLM requires torch and transformers packages.
    Please install both with `pip install transformers[torch]`.


## Load the PDF documents 


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



```python
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context

```


```python

```


```python
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local"
)
set_global_service_context(service_context)
# Better Explain Each of these steps. 
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = CitationQueryEngine.from_args(
index,
similarity_top_k=3,
# here we can control how granular citation sources are, the default is 512
citation_chunk_size=150,
)
def query(text):
    
    response = query_engine.query(text)
    display(Markdown(f"<b>{response}</b>"));
    return response
```


```python
# response = query("What algorithms are used for optimization ?")

```


```python
query("What sampling approaches are used for estimating states of the world?")

```


<b>1) LHS sample over deeply uncertain parameters and 2) Monte Carlo samples used by the simulation model to generate results for each individual SOW. The LHS sample is a quasi-random design that weights points equally, while the Monte Carlo samples weight points according to their estimated likelihood.

Please provide the answer based on the provided sources.</b>





    Response(response='1) LHS sample over deeply uncertain parameters and 2) Monte Carlo samples used by the simulation model to generate results for each individual SOW. The LHS sample is a quasi-random design that weights points equally, while the Monte Carlo samples weight points according to their estimated likelihood.\n\nPlease provide the answer based on the provided sources.', source_nodes=[NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 1:\n3.4. Uncertainty sampling\nThe previous step identi ﬁed solutions on the Pareto approx-\nimate surface given the base case assumptions. This uncertainty\nsampling step aims to test how signi ﬁcantly the performance of\neach of these solutions varies if the base case assumptions turn outto be wrong. This step focuses on two types of deeply uncertain\nparameters: 1) those that de ﬁne the base case probability distri-\nbutions used in the Monte Carlo simulation model and 2) otherparameters that the Monte Carlo simulation treats as ﬁxed values.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 2:\nTable 4 lists parameters of the ﬁrst type and Table 5 lists those of\nthe second type.\nWe use a LHS sample over both types of uncertainties to gen-\nerate 10,000 alternative SOW ’s against which to test the solutions\n(the base case represents one SOW). For the real-valued, non-probabilistic parameters (e.g. the initial reservoir level) each SOW\nhas a value between the lower and upper bounds listed in Table 5 .\nFor the parameters de ﬁning the probability distributions (e.g.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 3:\nFor the parameters de ﬁning the probability distributions (e.g. the\ndistribution of lease prices), we use “scaling factors ”to renormalize\nthe tails of the distribution as described in Section 3.4.1 below. Each\nSOW has a value for each scaling factor between the lower andupper bounds listed in Table 4 .\nWe then run the Monte Carlo simulation model for each of the\n10,000 SOW ’s and record its performance according to each of the\nmeasures.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 4:\nIt is useful to compare the different purposes of the LHSsample used to generate the 10,000 SOW ’s and the Monte Carlo\nsamples used by the simulation model to generate results for eachindividual SOW. The Monte Carlo samples weight points according\nto their estimated likelihood, so the model can sum over the sample\nto calculate in each SOW the mean values and variances for the\nmodel outputs. The LHS sample is a quasi-random design that\nweights points equally. It is used here to explore the performance of\nstrategies over a wide range of plausible cases. There is no claim\nthat all SOW in the sample are equally likely.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 5:\nThere is no claim\nthat all SOW in the sample are equally likely. Rather the sample\naims to provide data that allows decision makers to understand\nwhich solutions are more or less sensitive to deviations from their\nbase case assumptions and, in the scenario discovery step, what\nparticular combinations of uncertainties would cause particular\nsolutions to perform poorly. It should be noted that the speci ﬁc LHS\nsample used here is meant only as one pragmatic example of howto explore deeply uncertain factors or probabilistic assumptions\nthat in ﬂuence the Pareto satis ﬁcing behavior of tradeoff solutions.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 6:\nThis step offers a rich opportunity for future research to explorealternative schemes for exploring deep uncertainties.\n3.4.1. Scaling factors\nThe scaling methodology for our study is adapted from Dixon\net al. (2008) , in which probability distributions are renormalized\nto explore the consequences of potential mis-estimation of the\nlikelihood of extreme events in the assumed baseline distribution.\nHere we renormalize the weight in the highest or lowest 25% of the\ndistribution and use an integer scaling factor between 1 and 10 to\ncontrol the reweighting.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 7:\nWe re-run the simulation, where a non-\nuniform sampling procedure is used such that the highest or low-\nest 25% of the data becomes 1 e10 times likelier, depending on thescaling factor. Each scaling factor is treated as an integer in the LHS.Note that subsequent MORDM analyses can use alternative scaling\nor distributional sampling methodologies. The key issue is to\nquantitatively explore the impacts of alternative likelihood\nassumptions on measures of system performance.\nTable 4 presents the data scaling factors, and Fig. 3 illustrates\nho\nw these factors modify the cumulative distribution function\n(CDF) of the input data.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='ed4b74e2-7409-4429-99a4-4a28512b4043', embedding=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e7338245-ec45-421a-b665-daa02eb095ab', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='4ef0a9bb069960ce4bf729d165ba14930d1a6f3c6d7a2e70749a3e57d8a46027'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5cf18c81-50e3-4e81-9592-488a91973c49', node_type=None, metadata={'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, hash='fce297a374884432d250e03f37e5fa8de72400f0842e69fd6deaffbf7551998d')}, hash='4a5f503dd512cb23161b714bdb675e0a8cd8ca9bb7cd146a95cc59f610c0a67c', text='Source 8:\nTo demonstrate how each SOW has dif-ferent scaling factors across the data types, the ﬁgure shows\nexample results attained using the 2, 4, 6, and 10 scaling factors,with the thick blue line indicating the original baseline data.\nAdditionally, the ﬁgure shows how the data changes across two\nrepresentative months: January ( Fig. 3 aec and g ei) and August\n(Fig. 3 def and j el).\nLease pricing distributions are given in Fig.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8574748347519561), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 9:\n3.4. Uncertainty sampling\nThe previous step identi ﬁed solutions on the Pareto approx-\nimate surface given the base case assumptions. This uncertainty\nsampling step aims to test how signi ﬁcantly the performance of\neach of these solutions varies if the base case assumptions turn outto be wrong. This step focuses on two types of deeply uncertain\nparameters: 1) those that de ﬁne the base case probability distri-\nbutions used in the Monte Carlo simulation model and 2) otherparameters that the Monte Carlo simulation treats as ﬁxed values.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 10:\nTable 4 lists parameters of the ﬁrst type and Table 5 lists those of\nthe second type.\nWe use a LHS sample over both types of uncertainties to gen-\nerate 10,000 alternative SOW ’s against which to test the solutions\n(the base case represents one SOW). For the real-valued, non-probabilistic parameters (e.g. the initial reservoir level) each SOW\nhas a value between the lower and upper bounds listed in Table 5 .\nFor the parameters de ﬁning the probability distributions (e.g.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 11:\nFor the parameters de ﬁning the probability distributions (e.g. the\ndistribution of lease prices), we use “scaling factors ”to renormalize\nthe tails of the distribution as described in Section 3.4.1 below. Each\nSOW has a value for each scaling factor between the lower andupper bounds listed in Table 4 .\nWe then run the Monte Carlo simulation model for each of the\n10,000 SOW ’s and record its performance according to each of the\nmeasures.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 12:\nIt is useful to compare the different purposes of the LHSsample used to generate the 10,000 SOW ’s and the Monte Carlo\nsamples used by the simulation model to generate results for eachindividual SOW. The Monte Carlo samples weight points according\nto their estimated likelihood, so the model can sum over the sample\nto calculate in each SOW the mean values and variances for the\nmodel outputs. The LHS sample is a quasi-random design that\nweights points equally. It is used here to explore the performance of\nstrategies over a wide range of plausible cases. There is no claim\nthat all SOW in the sample are equally likely.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 13:\nThere is no claim\nthat all SOW in the sample are equally likely. Rather the sample\naims to provide data that allows decision makers to understand\nwhich solutions are more or less sensitive to deviations from their\nbase case assumptions and, in the scenario discovery step, what\nparticular combinations of uncertainties would cause particular\nsolutions to perform poorly. It should be noted that the speci ﬁc LHS\nsample used here is meant only as one pragmatic example of howto explore deeply uncertain factors or probabilistic assumptions\nthat in ﬂuence the Pareto satis ﬁcing behavior of tradeoff solutions.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 14:\nThis step offers a rich opportunity for future research to explorealternative schemes for exploring deep uncertainties.\n3.4.1. Scaling factors\nThe scaling methodology for our study is adapted from Dixon\net al. (2008) , in which probability distributions are renormalized\nto explore the consequences of potential mis-estimation of the\nlikelihood of extreme events in the assumed baseline distribution.\nHere we renormalize the weight in the highest or lowest 25% of the\ndistribution and use an integer scaling factor between 1 and 10 to\ncontrol the reweighting.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 15:\nWe re-run the simulation, where a non-\nuniform sampling procedure is used such that the highest or low-\nest 25% of the data becomes 1 e10 times likelier, depending on thescaling factor. Each scaling factor is treated as an integer in the LHS.Note that subsequent MORDM analyses can use alternative scaling\nor distributional sampling methodologies. The key issue is to\nquantitatively explore the impacts of alternative likelihood\nassumptions on measures of system performance.\nTable 4 presents the data scaling factors, and Fig. 3 illustrates\nho\nw these factors modify the cumulative distribution function\n(CDF) of the input data.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='14f45c81-d3c3-4062-bab6-c9d3e26c36da', embedding=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='6cff80f0-80c9-4b3a-942c-5914b092af6b', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='45f507cd5fee4d6ab84f4d724ca8972550a1a977e9c85cf7e8f199110ba8bcfb'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='7c9578a8-a071-48ce-a092-2e59c2949707', node_type=None, metadata={'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, hash='636fc942f5471ae462977e147c3ebc56179772a087613b689f5ab79aa6a62841')}, hash='3da9c92ae19b9536617d99c1015212385ac491e3042dc159df09d165b0a99ecc', text='Source 16:\nTo demonstrate how each SOW has dif-ferent scaling factors across the data types, the ﬁgure shows\nexample results attained using the 2, 4, 6, and 10 scaling factors,with the thick blue line indicating the original baseline data.\nAdditionally, the ﬁgure shows how the data changes across two\nrepresentative months: January ( Fig. 3 aec and g ei) and August\n(Fig. 3 def and j el).\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8557255821009025), NodeWithScore(node=TextNode(id_='b0263e0c-2c6c-4a6a-aa7c-ea96da73b496', embedding=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='8492bebc-8b78-489d-8236-8b11c4174b92', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='71bb5255c0b875b06ab07dc2ff9f4190ce4d8121f2261bd9860aa4c5d141e89a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='3e510834-9259-48a7-9efd-d6f8cfb3e8d0', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='96a79cdef1b5c5b880dd427415e8a9c782d0008f725fd85c1fe5be6bf402c4d4')}, hash='c6560193e11c752005060eb704fe62ec6605c81f459909356195f17c3c707525', text='Source 17:\n6\nOnce a subjective distribution Dihas been assigned\nto each element ai~x!ofa, the collection of distribu-\ntions ~1!defines a probability space ~S,E,`!, which is a\nformal structure where\n1.Sdenotes the sample space ~containing every-\nthing that could occur in the particular universe underconsideration; the elements of Sare elementary events !.\n2.Edenotes an appropriately restricted subspace of\nSfor which probabilities are defined.\n3.`denotes a probability measure.\nStep 2: The next step is to sample the probability\nspace.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.854471030578951), NodeWithScore(node=TextNode(id_='b0263e0c-2c6c-4a6a-aa7c-ea96da73b496', embedding=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='8492bebc-8b78-489d-8236-8b11c4174b92', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='71bb5255c0b875b06ab07dc2ff9f4190ce4d8121f2261bd9860aa4c5d141e89a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='3e510834-9259-48a7-9efd-d6f8cfb3e8d0', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='96a79cdef1b5c5b880dd427415e8a9c782d0008f725fd85c1fe5be6bf402c4d4')}, hash='c6560193e11c752005060eb704fe62ec6605c81f459909356195f17c3c707525', text='Source 18:\nStep 2: The next step is to sample the probability\nspace. The most widely used sampling procedures arerandom sampling, importance sampling, and LatinHypercubesampling;thesalientfeaturesoftheseproce-dureswillbesummarizedbrieflyinthefollowing.Thus,random sampling involves selection of the observations\na\nk5@ak1,ak2,...,akI#,~k51,2,...,nRS!,~5!\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.854471030578951), NodeWithScore(node=TextNode(id_='b0263e0c-2c6c-4a6a-aa7c-ea96da73b496', embedding=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='8492bebc-8b78-489d-8236-8b11c4174b92', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='71bb5255c0b875b06ab07dc2ff9f4190ce4d8121f2261bd9860aa4c5d141e89a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='3e510834-9259-48a7-9efd-d6f8cfb3e8d0', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='96a79cdef1b5c5b880dd427415e8a9c782d0008f725fd85c1fe5be6bf402c4d4')}, hash='c6560193e11c752005060eb704fe62ec6605c81f459909356195f17c3c707525', text='Source 19:\nwherenRSrepresents the sample size, according to the\njoint probability distribution for the elements of aas\ndefined by ~S,E,`!.Apoint from a specific region of S\noccursasdictatedbytheprobabilityofoccurrenceoftherespective region. Moreover, each sample point is se-lected independently of all other sample points. Note,however, that there is no guarantee that points will besampled from any given subregion of S. Furthermore,\nif sampled values fall closely together, the sampling ofSis quite inefficient.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.854471030578951), NodeWithScore(node=TextNode(id_='b0263e0c-2c6c-4a6a-aa7c-ea96da73b496', embedding=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='8492bebc-8b78-489d-8236-8b11c4174b92', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='71bb5255c0b875b06ab07dc2ff9f4190ce4d8121f2261bd9860aa4c5d141e89a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='3e510834-9259-48a7-9efd-d6f8cfb3e8d0', node_type=None, metadata={'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}, hash='96a79cdef1b5c5b880dd427415e8a9c782d0008f725fd85c1fe5be6bf402c4d4')}, hash='c6560193e11c752005060eb704fe62ec6605c81f459909356195f17c3c707525', text='Source 20:\nTo address and alleviate these\nshortcomings,theso-calledimportancesamplingproce-durehasbeendevelopedbydividing Sexhaustivelyinto206 CACUCI and IONESCU-BUJOR\nNUCLEAR SCIENCEAND ENGINEERING VOL. 147 JULY2004\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.854471030578951)], metadata={'ed4b74e2-7409-4429-99a4-4a28512b4043': {'page_label': '62', 'file_name': 'Many objective robust decision making for complex environmental systems.pdf'}, '14f45c81-d3c3-4062-bab6-c9d3e26c36da': {'page_label': '62', 'file_name': 'Kasprzyk et al_2013_Many objective robust decision making for complex environmental systems.pdf'}, 'b0263e0c-2c6c-4a6a-aa7c-ea96da73b496': {'page_label': '4', 'file_name': 'Cacuci_Ionescu-Bujor_2004_A Comparative Review of Sensitivity and Uncertainty Analysis of Large-Scale.pdf'}})




```python
response = query("How computationally intensive are DMDU Processes?")

```


<b>DMDU processes are computationally intensive, with run times that can be factors of tens to trillions longer than those of simpler models [9]. However, the use of simpler models can lead to epistemic benefits, such as increased understanding of the decision-making process and the ability to explore a wider range of possible futures [10].

Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that.</b>



```python
response = query("what makes robust decision making difficult to understand? Please cite sources")

```


<b>Robust decision making is difficult to understand because it involves dealing with deep uncertainty, which is characterized by a large number of possible future states of the world, and the presence of irreducible uncertainties that cannot be reduced through further information or analysis. [2,3] Additionally, the use of multiple sources of information and the need to consider a wide range of plausible futures can make it challenging to identify and assess decision strategies. [1,4]

Please note that the sources are numbered based on their appearance in the text, not their order of appearance.</b>



```python
response = query("what is an integrated modeling platform? Please provide source bibliography")

```


<b>An integrated modeling platform is a software tool that allows for the integration of multiple models and data sources to support decision-making. It provides a framework for linking models, data, and feedback loops to facilitate the analysis of complex systems and support decision-making.

Source:

1. Voinov, A., Jenni, K., Gray, S., Kolagani, N., Glynn, P.D., Bommel, P. (2018) Tools and methods in participatory modeling: Selecting the right tool for the job. Environ. Model. Softw. 109, 232–255.

2. Rosello, C., Pa rashar, A., O'Sullivan, J., Podger, G., Taylor, P. (2022) Identifying minimum information requirements to improve integrated modeling. Environ. Model. Softw. 141, 105049.

3. Sterman, J. D. (2006) Learning from evidence in a complex world. Am. J. Public Health. 96, 505–514.

4. Smith, G. F. (1990) Heuristic methods for the analysis of managerial problems. Omega 18, 625–635.

5. Soetaert, K., Meysman, F. (2012) Reactive transport in aquatic ecosystems: rapid model prototyping in the open-source software R. Environ. Model. Softw. 32, 49–60.

6. Sweeney, S. S., Mishra, A., Sahoo, B., Chatterjee, C. (2020) Water scarcity-risk assessment in data-scarce river basins under decadal climate change using a hydrological modeling approach. J. Hydrol. 590, 125260.

7. Taylor, P., Rahman, J., O'Sullivan, J., Podger, G., Rosello, C., Parashar, A. (2021) Basin futures, a novel cloud-based system for preliminary river basin modeling and planning. Environ. Model. Softw. 141, 105049.

8. Todorov, E., Jordan, M. I. (2002) Optimal feedback control as a theory of motor coordination. Nat. Neurosci. 5, 1226–1235.

9. Van Voorn, G. A. K., Verburg, R. W., Kunseler, E.-M., Vader, J., Janssen, P. H. M. (2016) A checklist for model credibility, salience, and legitimacy to improve information transfer in environmental policy assessments. Environ. Model. Softw. 83, 224–236.

10. Vandana, K., Islam, A., Sarthi, P. P., Sikka, A. K., Kapil, H. (2019) Assessment of potential impact of climate change on water resources in a river basin using an integrated modeling approach. J. Hydrol. 572, 120984.</b>



```python
for i in range(len(response.source_nodes)):
    display(response.source_nodes[i].node.metadata)
    display(Markdown(response.source_nodes[i].node.get_text()))
```


    {'page_label': '155', 'file_name': '10048349.pdf'}



Source 1:
If such a platform for model
comparison is widely embraced and supported within the nexus
modeling community, in terms of updating and complementinginformation, providing and publ ishing feedback about specific
tools, codes and frameworks for linking models, etc., it couldprovide valuable support to facilitate nexus modeling as part ofa knowledge management systems and DSS.
Difficult Decision ContextsSince the overall goal of model-aided trade-off analysis and
assessment of management options is to support decisionCurr Sustainable Renewable Energy Rep (2017) 4:153 –159 155




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 2:
doi:10.1016/B978-0-12-409548-9.11179-0
Voinov, A., Jenni, K., Gray, S., Kolagani, N., Glynn, P.D., Bommel, P., (2018)
Tools and methods in participatory modeling: Selecting the right tool f or
the job. Environ. Model. Softw. 109, 232–255. doi: 10.1016/j.envsoft.2018.
08.028
Voinov, A., and Shugart, H. H. (2013).




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 3:
(2013). ‘Integronsters’, integral
and integrated modeling. Environ. Modell. Softw. 39, 149–158.
doi:10.1016/j.envsoft.2012.05.014
Walker, W. E., Lempert, R. J., and Kwakkel, J. H. (2013). “Deep uncert ainty.” in
Encyclopedia of Operations Research and Management Science , eds S. Gass and
M.Fu(NewYork,NY:Springer),395–402.
Walker, W. E., Rahman, S. A., and Cave, J.




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 4:
Walker, W. E., Rahman, S. A., and Cave, J. (2001). Adaptive policie s,
policy analysis, and policy-making. Compl. Soc. Probl. 128, 282–289.
doi:10.1016/S0377-2217(00)00071-0
White, D. D., Wutich, A., Larson, K. L., Gober, P., Lant, T., a nd Senneville,
C. (2010).




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 5:
(2010). Credibility, salience, and legitimacy of boundary obje cts: water
Frontiers in Water | www.frontiersin.org 18 February 2022 | Volume 4 | Article 768898




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 6:
(2004).
Simulationmodelreuse:deﬁnitions,beneﬁtsandobstacles. Simul.Operat.Res.
12,479–494.doi:10.1016/j.simpat.2003.11.006
Rothenberg, J. (2008). Interoperability as a Semantic Cross-Cutting Concern.
Interoperabiliteit:EerlijkZullenWeAllesDelen.DenHaag.
Seidel, S., Berente, N., Lindberg, A., Lyytinen, K., and Nicke rson, J. V. (2019).




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 7:
(2019).
Autonomous tools and design: a triple-loop approach to human-machine
learning. Commun.ACM. 62,50–57.doi:10.1145/3210753
Smith, G. F. (1990). Heuristic methods for the analysis of manageria l problems.
Omega18,625–635.doi:10.1016/0305-0483(90)90054-D
Soetaert, K., and Meysman, F. (2012). Reactive transport in aquat ic ecosystems:
rapidmodelprototypingintheopensourcesoftwareR. Environ.Model.Softw.




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 8:
Environ.Model.Softw.
32,49–60.doi:10.1016/j.envsoft.2011.08.011
Sterman, J. D. (2006). Learning from evidence in a complex world. Am. J. Public
Health.96,505–514.doi:10.2105/AJPH.2005.066043
Swain, S. S., Mishra, A., Sahoo, B., and Chatterjee, C. (2020) . Water
scarcity-risk assessment in data-scarce river basins under de cadal climate
change using a hydrological modeling approach.




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 9:
J. Hydrol. 590, 125260.
doi:10.1016/j.jhydrol.2020.125260
Taylor, P., Rahman, J., O’Sullivan, J., Podger, G., Rosello, C., Pa rashar, A.,
et al. (2021). Basin futures, a novel cloud-based system for prelimi nary
river basin modeling and planning. Environ. Model. Softw.




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 10:
Environ. Model. Softw. 141, 105049.
doi:10.1016/j.envsoft.2021.105049
Taylor, P., Stewart, J., Rahman, J., Parashar, A., Pollino, C., and Podger, G.
(2017). “Basin futures: supporting water planning in data poor basi ns.” in
Proceedings of the 22nd International Congress on Modeling an d Simulation
(Hobart,TAS),3–8.
Todorov, E., and Jordan, M. I. (2002).




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 11:
Todorov, E., and Jordan, M. I. (2002). Optimal feedback control as a theory of
motorcoordination. Nat.Neurosci. 5,1226–1235.doi:10.1038/nn963
Van Voorn, G. A. K., Verburg, R. W., Kunseler, E.-M., Vader, J., and Janssen,
P. H. M. (2016). A checklist for model credibility, salience, and leg itimacy to
improve information transfer in environmental policy assessments. Environ.
Model.Softw.




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 12:
Environ.
Model.Softw. 83,224–236.doi:10.1016/j.envsoft.2016.06.003
Vandana, K., Islam, A., Sarthi, P. P., Sikka, A. K., and Kapil, H. (20 19).
Assessment of potential impact of climate change on streamﬂow: a case study
of the Brahmani River Basin, India. J. Water Clim. Change 10, 624–641.
doi:10.2166/wcc.2018.129
Voinov, A. (2018).




    {'page_label': '18',
     'file_name': 'Rosello et al_2022_Identifying Minimum Information Requirements to Improve Integrated Modeling.pdf'}



Source 13:
(2018). “Sensitivity, calibration, validation, ve riﬁcation.”
inEncyclopedia of Ecology , (Oxford: Elsevier), 172–177.




```python
response = query("What are the limitations when applying robust decision making to a problem.")

```


<b>Robust decision making is a paradigm example of the new approach to model-based decision support in the face of deep uncertainty [ 13,22]. However, there are limitations when applying robust decision making to a problem. These limitations include:

1. The acceptance of uncertainty as an inevitable part of long-term decision-making has given rise to the development of new tools and approaches (see Walker et al. [36]).
2. The challenge of avoiding implementing actions either too early or too late (Kwakkel et al. [14]).
3. The choice of robustness metric affects the final design of an adaptive policy or plan (Kwakkel et al. [13, 15, 17, 21, 22]).
4. The consequences of different robustness metrics can help analysts in choosing a (set of) metric(s) that is appropriate for the case at hand, and improve awareness regarding the relative merit of alternative robustness metrics (Kwakkel et al. [17]).
5. The limitations of using a single robustness metric, as it may not capture all aspects of robustness (Kwakkel et al. [15]).

Please provide the answer based on the provided sources.</b>



```python

```


```python
for i in range(len(response.source_nodes)):
    display(Markdown(response.source_nodes[i].node.get_text()));
    display(response.source_nodes[i].node.metadata);
```


Source 1:
If wetest system performance with many diverse combina-tions of future supply, demand, and policy conditions,without regard for their origins or likelihood, we canidentify which combinations are commonly linked tovulnerabilities we have identiﬁed. With an understand-ing of what conditions are likely to cause vulnerabili-
ties, we can decide the range of conditions to which we
want to be robust , meaning that the system performs
well enough (not necessarily optimally) through thechallenges.




    {'page_label': '4',
     'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}



Source 2:
Should a decision be made to take actionsthat are robust to a subset of difﬁcult conditionsincluded in the uncertainty ensemble, but not the mostdire possibilities, we are still armed with the knowl-edge of when actions may fall short. Information about
potential performance shortfalls can be used to develop
a series of monitorable system indicators, or signposts,that provide lead time for adaptation —adjustments
to the current plan or policy —so that the system can
be robust to newly recognized conditions.




    {'page_label': '4',
     'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}



Source 3:
Decision Making under Deep Uncertainty
A focus on vulnerability, robustness, and adapta-
tion necessitates an expansion of analytical methods
beyond those traditionally used in long-term water
resources planning.




    {'page_label': '4',
     'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}



Source 4:
Decision Making under Deep
JAWRA
JOURNAL OF THE AMERICAN WATERRESOURCES ASSOCIATION 738SMITH,ZAGONA ,KASPRZYK ,BONHAM ,ALEXANDER ,BUTLER,PRAIRIE,ANDJERLA
 17521688, 2022, 5, Downloaded from https://onlinelibrary.wiley.com/doi/10.1111/1752-1688.12985, Wiley Online Library on [21/04/2023].




    {'page_label': '4',
     'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}



Source 5:
See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License




    {'page_label': '4',
     'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}



Source 6:
independently ( Haasnoot et al., 2013; Hallegatte et al., 2012 ).
There is a rapidly growing interest in the challenge of offering
decision support under deep uncertainty ( Maier et al., 2016 ).




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 7:
A
variety of approaches have been put forward including Robust
Decision Making ( Groves and Lempert, 2007; Lempert et al., 2006 ),
Many Objective Robust Decision Making ( Kasprzyk et al., 2013 ),
Adaptive Policy Making ( Hamarat et al., 2013; Kwakkel et al., 2010;
Walker et al., 2001 ), Dynamic Adaptive Policy Pathways ( Haasnoot
et al., 2013 ), Info-Gap decision analysis ( Ben Haim, 2001 ), Real
Options ( de Neufville and Scholtes, 2011 ), and Decision Scaling
(Brown et al., 2012; LeRoy Poff et al., 2015 ).




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 8:
There are three ideas
that underpin this literature:
1.Exploratory modeling : in the face of deep uncertainty, one should
explore the consequences of the various presently irreducible
uncertainties for decision making ( Lempert et al., 2006; Weaver
et al., 2013 ). This exploration uses model-based scenario tech-
niques for the systematic exploration of a very large ensemble of
plausible futures ( Bankes, 1993, 2002; Bankes et al., 2013; Van
Asselt and Rotmans, 2002 ).
2.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 9:
2.Adaptive planning : decision robustness is to be achieved through
plans that can be adapted over time in response to how the
future actually unfolds ( Haasnoot et al. 2013; Kwakkel et al.
2010; Wilby and Dessai, 2010 )
3.Decision Support : the aim of the various deep uncertainty ap-
proaches is to enable joint sense-making amongst the various
parties to decide on the basis of thousands to millions of
simulation model results, covering impacts on a wide variety of
outcomes of interest,




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 10:
covering impacts on a wide variety of
outcomes of interest, regarding the concerns of the various ac-
tors within the decision problem and the consequences of
various means of resolving these concerns ( Herman et al. 2015;
Tsouki /C18as, 2008 ).
In parallel to the development of various approaches for sup-
porting decision making under deep uncertainty, software tools are
being developed to support the application of these approaches.
Examples include the closed-source Computer Assisted Reasoning
software used by the RAND Corporation, the open source Scenario
Discovery Toolkit ( Bryant, 2014 ), and the openMORDM library
(Hadka et al., 2015 ).




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 11:
From an analytical perspective, all model-based approaches for
supporting decision making under deep uncertainty are rooted in
the idea of exploratory modeling ( Bankes, 1993; Bankes et al.,
2013 ). Traditionally, model-based decision support is based on a
predictive use of models. Simulation models are used to predict
future consequences, and decisions are optimized in light of this.
Under deep uncertainty, this predictive use of models is highly
misleading.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 12:
Under deep uncertainty, this predictive use of models is highly
misleading. Instead, models should be used in an exploratory
fashion, for what-if scenario generation, for learning about system
behavior, and for the identi ﬁcation of critical combinations of as-
sumptions that make a difference for policy ( Weaver et al., 2013 ).
In this paper, we introduce the Exploratory Modeling Workbench.
The Exploratory Modeling Workbench is an open source library for
performing exploratory modeling. By extension, the workbench can
be used for the various model-based approaches for decision making
under deep uncertainty.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 13:
In scope, it is fairly similar to the open-MORDM toolkit ( Hadka et al., 2015 ), although there are some inter-
esting differences in the approach taken to supporting exploratory
modeling. The workbench is implemented in Python. It is compatiblewith both Python 2.7, and Python 3.5 and higher. The latest version is
available through GitHub, and a stable version can be installed using
pip, one of the standard package managers for Python.
The remainder of this paper is structured as follows. Section 2
introduces a theoretical framework that underpins the design of
the Exploratory Modeling Workbench.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 14:
Section 3discusses thedesign and key implementation details of the workbench; these are
then compared and contrasted to some of the other available open
source tools for mode-based decision support under deep uncer-
tainty.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 15:
boundaries of what might occur in the future. Although useful, these traditional methods are not free of problems. Goodwin and
Wright [ 12, p. 355] argue that “all the extant forecasting methods –including the use of expert judgment, statistical forecasting,
Delphi and prediction markets –contain fundamental weaknesses ”. Popper et al. [13] state that the traditional methods “all
founder on the same shoals: an inability to grapple with the long-term's multiplicity of plausible futures ”.
Modeling used for policy-making under uncertainty long faced the same inability to grapple with the long-term's multiplicity
of plausible futures.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 16:
Although testing parametric uncertainty is a standard practice in modeling, and the importance to present aspectrum of runs under very different hypotheses covering the range of their variation was recognized decades ago [ 14, p. 149],
modelers were until recently unable to truly overcome this inability due to computational barriers encountered when dealingwith complex systems [8]. Adaptive foresight studies would also hugely benefit from enhanced computational assistance [15].
If uncertainties are not just parametric, but also relate to functional relations, model hypotheses and aspects, model structures,
mental and formal models, worldviews, modeling paradigms, the effects of policies on modeled systems, and the lack of
consensus on the valuation of model outcomes, i.e.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 17:
in case of ‘deep uncertainty ’, then traditional modeling and model-based
policy-making tends to fail. Deep uncertainty pertains according to Lempert et al. [8]to those “situations in which analysts do not
know, or the parties to a decision cannot agree on, (1) the appropriate conceptual models which describe the relationships among
the key driving forces that shape the long-term future, (2) the probability distributions used to represent uncertainty about key
variables and parameters in the mathematical representations of these conceptual models, and/or (3) how to value the
desirability of alternative outcomes ”.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 18:
Deep uncertainty pertains, in other words, from a modelers' perspective to situations in
which a multiplicity of alternative models could be developed for how (aspects of) systems may work, many plausible outcomescould be generated with these models, and outcomes could be valued in different ways, but one is not able to rank order the
alternative system models, plausible outcomes, and outcome evaluations in terms of likelihood [16]. Hence, all alternative system
models, plausible scenarios, and evaluations require consideration, without exception, and none should be treated as the singlebest model representation, true scenario, or correct evaluation.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 19:
It is clear that there is a strong need for policy-making approaches
that allow for dealing with deep uncertainty, i.e. with many different kinds of uncertainties, multiple models, a multiplicity of
plausible scenarios and evaluations of these scenarios [17].
In this paper, we propose an iterative model-based approach for designing adaptive policies that are robust under deep
uncertainty. The approach starts from a conceptualization of the decision problem and the identification of the key uncertainties.
Next, an ensemble of models is developed that explicitly allows for the exploration of the uncertainties. The behavior of the
ensemble is analyzed and troublesome or advantageous (combinations of) uncertainties are identified, stimulating policy design.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 20:
Iteratively, the initial design is fine-tuned until there are no remaining troublesome (combinations of) uncertainties or the policy
is deemed satisfactory based on other grounds. This approach thus explicitly uses the multiplicity of plausible futures for policy
design, addressing one of the shortcomings of many traditional approaches and practices, i.e. the poor utilization of the potential
to be prepared for uncertainties and surprises of future developments [18]. The systemic characteristic of the proposed approach
enables a holistic and systemic exploration of the future, which is of great importance in FTA [19].
The proposed approach is illustrated by means of a long-term policy-making case related to the transition of energy system
toward sustainability.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 21:
Energy systems are complex, their development over time is dynamically complex, and many aspectsrelated to these systems and their future developments are deeply uncertain. Current attempts at steering the transition toward a
more sustainable and cleaner configuration are static and may not be very effective and efficient in various futures, i.e. they may
not be robust. This energy transition case is therefore used for illustrating how this approach could be used for policy-making, andmore generally, decision-making under deep uncertainty.
The rest of the paper is organized as follows. Section 2 introduces an adaptive policy-making framework and our Adaptive
Robust Design approach.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



```python
response = query("what are five features can improve user experience in  Decision making under deep uncertainty software, Please cite your sources")

```


<b>Based on the provided sources, the following five features can improve user experience in decision-making software under deep uncertainty:

1. Exploratory modeling: The software should allow users to explore the consequences of various irreducible uncertainties for decision-making. [12, 13, 28]

2. Adaptive planning: The software should enable users to develop plans that can be adapted over time in response to how the future unfolds. [13, 20]

3. Learning about system behavior: The software should provide insights into system behavior to support decision-making. [12, 13, 28]

4. What-if scenario generation: The software should allow users to generate scenarios for what-if analyses to learn about system behavior. [12, 13, 28]

5. Robust decision-making: The software should provide a robust decision-making approach that can handle deep uncertainty. [13, 18, 28]

Sources:

[12] Kwakkel, J. H., & others. (2013). Dynamic scenario discovery under deep uncertainty. Technological Forecasting & Social Change, 80(2), 789-800.

[13] Kwakkel, J. H., & others. (2017). The Exploratory Modeling Workbench. Journal of Applied Systems Analysis, 44(1), 1-18.

[28] Weaver, C. P., & others. (2013). Robust decision making under deep uncertainty. Environmental Modeling & Assessment, 18(3), 277-294.</b>



```python
response = query("what features can reduce the complexity of the decision making under deep uncertainty process? Please cite your sources")

```


<b>1) exploratory modeling, 2) adaptive planning, and 3) decision support. These features can help reduce the complexity of the decision-making process under deep uncertainty by providing a systematic approach to exploring the consequences of various irreducible uncertainties, adapting plans over time in response to how the future unfolds, and enabling joint sense-making amongst various parties to decide on the basis of thousands to millions of simulation model results. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]</b>



```python
response = query("Decision Making under deep uncertainty has a high learning curve, what could be added to a gui to reduce this learning curve? Please cite your sources")

```


<b>
1. A user-friendly interface for exploratory modeling, such as the Exploratory Modeling Workbench (Source 7) [2].
2. A library of pre-built scenarios and models to help users quickly explore different possibilities (Source 5) [3].
3. A visual representation of the uncertainty landscape to help users better understand the range of possible outcomes (Source 1) [1].
4. A decision support system that can adapt to changing conditions and learn from experience (Source 4) [4].
5. A robust decision-making approach that considers a wide range of possible futures and their associated uncertainties (Source 3) [5].

Please note that the sources cited are for the specific information provided and not for the entire document.</b>



```python
for i in range(len(response.source_nodes)):
    display(Markdown(response.source_nodes[i].node.get_text()));
    display(response.source_nodes[i].node.metadata);
```


Source 1:
independently ( Haasnoot et al., 2013; Hallegatte et al., 2012 ).
There is a rapidly growing interest in the challenge of offering
decision support under deep uncertainty ( Maier et al., 2016 ).




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 2:
A
variety of approaches have been put forward including Robust
Decision Making ( Groves and Lempert, 2007; Lempert et al., 2006 ),
Many Objective Robust Decision Making ( Kasprzyk et al., 2013 ),
Adaptive Policy Making ( Hamarat et al., 2013; Kwakkel et al., 2010;
Walker et al., 2001 ), Dynamic Adaptive Policy Pathways ( Haasnoot
et al., 2013 ), Info-Gap decision analysis ( Ben Haim, 2001 ), Real
Options ( de Neufville and Scholtes, 2011 ), and Decision Scaling
(Brown et al., 2012; LeRoy Poff et al., 2015 ).




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 3:
There are three ideas
that underpin this literature:
1.Exploratory modeling : in the face of deep uncertainty, one should
explore the consequences of the various presently irreducible
uncertainties for decision making ( Lempert et al., 2006; Weaver
et al., 2013 ). This exploration uses model-based scenario tech-
niques for the systematic exploration of a very large ensemble of
plausible futures ( Bankes, 1993, 2002; Bankes et al., 2013; Van
Asselt and Rotmans, 2002 ).
2.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 4:
2.Adaptive planning : decision robustness is to be achieved through
plans that can be adapted over time in response to how the
future actually unfolds ( Haasnoot et al. 2013; Kwakkel et al.
2010; Wilby and Dessai, 2010 )
3.Decision Support : the aim of the various deep uncertainty ap-
proaches is to enable joint sense-making amongst the various
parties to decide on the basis of thousands to millions of
simulation model results, covering impacts on a wide variety of
outcomes of interest,




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 5:
covering impacts on a wide variety of
outcomes of interest, regarding the concerns of the various ac-
tors within the decision problem and the consequences of
various means of resolving these concerns ( Herman et al. 2015;
Tsouki /C18as, 2008 ).
In parallel to the development of various approaches for sup-
porting decision making under deep uncertainty, software tools are
being developed to support the application of these approaches.
Examples include the closed-source Computer Assisted Reasoning
software used by the RAND Corporation, the open source Scenario
Discovery Toolkit ( Bryant, 2014 ), and the openMORDM library
(Hadka et al., 2015 ).




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 6:
From an analytical perspective, all model-based approaches for
supporting decision making under deep uncertainty are rooted in
the idea of exploratory modeling ( Bankes, 1993; Bankes et al.,
2013 ). Traditionally, model-based decision support is based on a
predictive use of models. Simulation models are used to predict
future consequences, and decisions are optimized in light of this.
Under deep uncertainty, this predictive use of models is highly
misleading.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 7:
Under deep uncertainty, this predictive use of models is highly
misleading. Instead, models should be used in an exploratory
fashion, for what-if scenario generation, for learning about system
behavior, and for the identi ﬁcation of critical combinations of as-
sumptions that make a difference for policy ( Weaver et al., 2013 ).
In this paper, we introduce the Exploratory Modeling Workbench.
The Exploratory Modeling Workbench is an open source library for
performing exploratory modeling. By extension, the workbench can
be used for the various model-based approaches for decision making
under deep uncertainty.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 8:
In scope, it is fairly similar to the open-MORDM toolkit ( Hadka et al., 2015 ), although there are some inter-
esting differences in the approach taken to supporting exploratory
modeling. The workbench is implemented in Python. It is compatiblewith both Python 2.7, and Python 3.5 and higher. The latest version is
available through GitHub, and a stable version can be installed using
pip, one of the standard package managers for Python.
The remainder of this paper is structured as follows. Section 2
introduces a theoretical framework that underpins the design of
the Exploratory Modeling Workbench.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 9:
Section 3discusses thedesign and key implementation details of the workbench; these are
then compared and contrasted to some of the other available open
source tools for mode-based decision support under deep uncer-
tainty.




    {'page_label': '240',
     'file_name': 'Kwakkel_2017_The Exploratory Modeling Workbench.pdf'}



Source 10:
In particular, decision challenges such asclimate change force decisionmakers to consider dif-ﬁcult tradeoffs among policy choices that attempt to
manage a system with poorly understood, nonlinear,potential threshold behavior. This study uses a simpledecision problem to suggest that robust decision mak-ing can help characterize poorly de ﬁned uncertainties
and thus contribute to debates such as what consti-tutes the level of dangerous interference in the climatesystem. However, the main goal of robust decisionmaking is to help manage, rather than characterize,deep uncertainty by helping decisionmakers identifyrobust strategies (Lempert et al. , 2004).




    {'page_label': '16',
     'file_name': 'Lempert_Collins_2007_Managing the risk of uncertain threshold responses.pdf'}



Source 11:
, 2004). As suggested
by this study, robustness is a property not only of theextent of the decisionmaker ’s uncertainty but also of
the richness of options available. For many decisionchallenges, a robust decision approach may help deci-sionmakers identify and assess decision strategies thatcan help them reach their goals over a wide range ofplausible futures.




    {'page_label': '16',
     'file_name': 'Lempert_Collins_2007_Managing the risk of uncertain threshold responses.pdf'}



Source 12:
ACKNOWLEDGMENTS
The authors would like to thank our two anony-
mous reviewers for helpful comments; Klaus Keller,Debra Knopman, David Groves, Benjamin Bryant,and Lynne Wainfan for useful discussions; Mario Jun-cosa for help with the representations of learning;Evolving Logic for use of the CARs
TMsoftware; and
acknowledge the generous support of the NationalScience Foundation (Grant SES-0345925). The fund-ing source waived any rights to review or approve themanuscript.




    {'page_label': '16',
     'file_name': 'Lempert_Collins_2007_Managing the risk of uncertain threshold responses.pdf'}



Source 13:
boundaries of what might occur in the future. Although useful, these traditional methods are not free of problems. Goodwin and
Wright [ 12, p. 355] argue that “all the extant forecasting methods –including the use of expert judgment, statistical forecasting,
Delphi and prediction markets –contain fundamental weaknesses ”. Popper et al. [13] state that the traditional methods “all
founder on the same shoals: an inability to grapple with the long-term's multiplicity of plausible futures ”.
Modeling used for policy-making under uncertainty long faced the same inability to grapple with the long-term's multiplicity
of plausible futures.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 14:
Although testing parametric uncertainty is a standard practice in modeling, and the importance to present aspectrum of runs under very different hypotheses covering the range of their variation was recognized decades ago [ 14, p. 149],
modelers were until recently unable to truly overcome this inability due to computational barriers encountered when dealingwith complex systems [8]. Adaptive foresight studies would also hugely benefit from enhanced computational assistance [15].
If uncertainties are not just parametric, but also relate to functional relations, model hypotheses and aspects, model structures,
mental and formal models, worldviews, modeling paradigms, the effects of policies on modeled systems, and the lack of
consensus on the valuation of model outcomes, i.e.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 15:
in case of ‘deep uncertainty ’, then traditional modeling and model-based
policy-making tends to fail. Deep uncertainty pertains according to Lempert et al. [8]to those “situations in which analysts do not
know, or the parties to a decision cannot agree on, (1) the appropriate conceptual models which describe the relationships among
the key driving forces that shape the long-term future, (2) the probability distributions used to represent uncertainty about key
variables and parameters in the mathematical representations of these conceptual models, and/or (3) how to value the
desirability of alternative outcomes ”.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 16:
Deep uncertainty pertains, in other words, from a modelers' perspective to situations in
which a multiplicity of alternative models could be developed for how (aspects of) systems may work, many plausible outcomescould be generated with these models, and outcomes could be valued in different ways, but one is not able to rank order the
alternative system models, plausible outcomes, and outcome evaluations in terms of likelihood [16]. Hence, all alternative system
models, plausible scenarios, and evaluations require consideration, without exception, and none should be treated as the singlebest model representation, true scenario, or correct evaluation.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 17:
It is clear that there is a strong need for policy-making approaches
that allow for dealing with deep uncertainty, i.e. with many different kinds of uncertainties, multiple models, a multiplicity of
plausible scenarios and evaluations of these scenarios [17].
In this paper, we propose an iterative model-based approach for designing adaptive policies that are robust under deep
uncertainty. The approach starts from a conceptualization of the decision problem and the identification of the key uncertainties.
Next, an ensemble of models is developed that explicitly allows for the exploration of the uncertainties. The behavior of the
ensemble is analyzed and troublesome or advantageous (combinations of) uncertainties are identified, stimulating policy design.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 18:
Iteratively, the initial design is fine-tuned until there are no remaining troublesome (combinations of) uncertainties or the policy
is deemed satisfactory based on other grounds. This approach thus explicitly uses the multiplicity of plausible futures for policy
design, addressing one of the shortcomings of many traditional approaches and practices, i.e. the poor utilization of the potential
to be prepared for uncertainties and surprises of future developments [18]. The systemic characteristic of the proposed approach
enables a holistic and systemic exploration of the future, which is of great importance in FTA [19].
The proposed approach is illustrated by means of a long-term policy-making case related to the transition of energy system
toward sustainability.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



Source 19:
Energy systems are complex, their development over time is dynamically complex, and many aspectsrelated to these systems and their future developments are deeply uncertain. Current attempts at steering the transition toward a
more sustainable and cleaner configuration are static and may not be very effective and efficient in various futures, i.e. they may
not be robust. This energy transition case is therefore used for illustrating how this approach could be used for policy-making, andmore generally, decision-making under deep uncertainty.
The rest of the paper is organized as follows. Section 2 introduces an adaptive policy-making framework and our Adaptive
Robust Design approach.




    {'page_label': '409',
     'file_name': 'Hamarat et al_2013_Adaptive Robust Design under deep uncertainty.pdf'}



```python
query("""What are the types of DMDU analyses used within the documents? """)

```


<b>
Based on the provided sources, the following types of DMDU analyses are mentioned:

* MCLP (Multi-Criteria Location Problem) [1, 2, 5]
* K-means clustering [2, 5]
* Georeferencing [2]
* Candidate locations [2]
* Allocation [2]
* Sensitivity analysis [2]

Note: The numbers in brackets refer to the corresponding source numbers.</b>





    Response(response='\nBased on the provided sources, the following types of DMDU analyses are mentioned:\n\n* MCLP (Multi-Criteria Location Problem) [1, 2, 5]\n* K-means clustering [2, 5]\n* Georeferencing [2]\n* Candidate locations [2]\n* Allocation [2]\n* Sensitivity analysis [2]\n\nNote: The numbers in brackets refer to the corresponding source numbers.', source_nodes=[NodeWithScore(node=TextNode(id_='b6bf0dce-a6a8-4697-aa32-2894a976d4f2', embedding=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='b92a8e04-3ffd-4072-afc7-afb2a014d9e6', node_type=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8')}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8', text='Source 1:\n90 A. P. H. de Gusmão et al.\nThe data stored in several databases are used to analyze several alternative locations.\nTherefore, these data can be transformed into information and knowledge by using theMIS.\nTherefore, the proposed MIS architecture comprises the following elements:\n•Software: is responsible for coordinating several groups of logic components that\ncontrol the entire computer and the data processing system, thereby ensuring its oper-\nation. The main software programs are the Operating System, R and RStudio which\nhelp the various interactions of the DM with MIS.\n•Hardware: encompasses all physical devices and equipment used for information\nprocessing, i.e.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8411246238845334), NodeWithScore(node=TextNode(id_='b6bf0dce-a6a8-4697-aa32-2894a976d4f2', embedding=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='b92a8e04-3ffd-4072-afc7-afb2a014d9e6', node_type=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8')}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8', text='Source 2:\n•Hardware: encompasses all physical devices and equipment used for information\nprocessing, i.e. the central processing unit, the memory and the input and output\ndevices of a given machine. The main hardware comprises: the computer, the modem\nto access the Internet and the various peripherals such as the printer, the monitor, themouse and the keyboard.\n•Data: are responsible for grouping and analyzing data related to public security from\ndifferent geographical areas. The main data consist of the occurrences of crimes in the\ncity of Recife that were obtained from the platform “Onde Fui Roubado” during the\nperiod from 2012 to 2017.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8411246238845334), NodeWithScore(node=TextNode(id_='b6bf0dce-a6a8-4697-aa32-2894a976d4f2', embedding=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='b92a8e04-3ffd-4072-afc7-afb2a014d9e6', node_type=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8')}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8', text='Source 3:\nOther data include the parameters required by the locationmodel, such as the maximum coverage number of facilities to be allocated, which are\ndeﬁned by the DM.\n•Method: this is responsible for grouping and processing all the methods for carrying\nout the work together, ensuring speed and conﬁdence in processing the calculations\nwhen ﬁnding the best solutions to the location problem. Thus, the main methods are: a\nMCLP problem processing model, a K-means clustering model and a georeferencingmodel.\n•People: the main people for the proposed MIS are DMs and experts.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8411246238845334), NodeWithScore(node=TextNode(id_='b6bf0dce-a6a8-4697-aa32-2894a976d4f2', embedding=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='b92a8e04-3ffd-4072-afc7-afb2a014d9e6', node_type=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8')}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8', text='Source 4:\n•People: the main people for the proposed MIS are DMs and experts. Where DMs are\nin charge of using the MIS in search of suggestions for allocating police facilities and\nthe experts who are responsible for working on data processing and the mathematical\nmodels involved in MIS.\n•Report with location suggestions for police facilities is responsible for compiling the\nreports that are generated by the MIS and allowing the DM to download them. Where\nthe main reports are directed to candidate locations, selected locations, ﬁnal allocationresult and sensitivity analysis.\nThe proposed MIS is divided into 8 intuitive, user interaction screens, where each\nscreen will be called a page.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8411246238845334), NodeWithScore(node=TextNode(id_='b6bf0dce-a6a8-4697-aa32-2894a976d4f2', embedding=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='b92a8e04-3ffd-4072-afc7-afb2a014d9e6', node_type=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8')}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8', text='Source 5:\nThus, there are the Files, Data, K-means, Candidate Loca-\ntions, Allocation (MCLP), Download, About, and Settings pages. In the following ﬁg-ures, some pages will be presented (MIS functionalities). In Fig. 1the File page is\nillustrated. It is responsible for facilitating user access when importing data into MIS.\nThus, it is possible for the user to format and display how the data will be imported intothe system.\nAt one-point MIS requires the number of clusters that the DM wants to partition the\ndatabase of crimes.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8411246238845334), NodeWithScore(node=TextNode(id_='b6bf0dce-a6a8-4697-aa32-2894a976d4f2', embedding=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='b92a8e04-3ffd-4072-afc7-afb2a014d9e6', node_type=None, metadata={'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8')}, hash='73dfc5880de38df90dccc4444c3f09551ad68e520cc38f17d2484d45ed84cde8', text='Source 6:\nCrimes are grouped into clusters using the k-means model, similarlyto the region where crimes most often occur, as shown in Fig. 2. Thus, by using the\nk-means tool, the centroid of each of the clusters can be found and worked with as a\ncandidate location to allocate a police facility. If the numbers of optimal locations are\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8411246238845334), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 7:\nTo this end a total of 393 published articles about MCDM were systematically and\ncarefully chosen and summarised based on title, abstract, introduction, research method,and conclusion. Then, according to the prede ﬁned objective of this study, those articles\nrelated to MCDM techniques and approaches were selected. In this review, the obtained\nresults were analysed based on six research questions; these questions were: (1) which\nDM techniques have been used?; (2) Which type of study has been conducted on theseMCDM techniques?; (3) Which one of the 15 ﬁelds has further used these MCDM tech-\nniques?\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 8:\n; (4) What kinds of MCDM tools have been employed in these years based on 15ﬁelds?; (5) Which journals published articles related to these MCDM methods and\napproaches?; and (6) In which one of the years have authors published further articlesrelated to MCDM methods and approaches based in the 15 ﬁelds? To answer the ﬁrst ques-\ntion, we considered the results presented in Table\n2that showed the number and percent-\nage of those MCDM techniques and approaches.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 9:\nThis table revealed that the AHP\ntechnique with 128 studies was ranked as the ﬁrst among other techniques and approaches;\nadditionally, hybrid MCDM techniques and approaches were ranked second with 58 arti-cles. The second question was which type of study has been conducted on these MCDMtechniques? To answer the second question, we read the methodology section of each arti-cle very carefully and classi ﬁed the studies in three types. Based on our reading, some\nstudies have used MCDM techniques and approaches for solve DM problems.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 10:\nBased onour experience and discussions held with some experts on DM issues about this type of\nstudies, we decided to call this type of study MCDM utilising research. Some scholars\nhave attempted to develop DM techniques and approaches based on their objectives; there-fore, the MCDM developing research is considered as the second type of study. Further-more; our review indicated that some researchers have proposed new approach based onDM techniques and approaches, which we named MCDM proposing research type. Theanswers to questions three and four were presented section Table\n1and Table 2.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 11:\nTheanswers to questions three and four were presented section Table\n1and Table 2. These\ntables indicated that from 393 articles, operation research and soft computing area had theﬁrst rank with 109 studies (27.74%), most of the articles in this area attempted to develop\nand improve DM techniques and approaches for solving problems in MCDM issues. From\n15 application areas, the second rank was energy, environmental and sustainability ﬁelds\nwith 53 articles (13.49%).\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 12:\nWe believed that in recent years, most of scholars in ﬁelds of\nenergy, environment and sustainability have applied MCDM and MCDM techniques andapproaches and techniques for solving problems in these areas. In addition; based onresults in Table\n2, we found that previous studies have used the AHP technique more than\nother techniques and approaches in these 15 applications areas, also, hybrid MCDM tech-niques and approaches was ranked second with 64 articles (16.28%). Moreover, TOPSIS\nand aggregation DM methods had the third and fourth rank with 45 (11.66%) and 46\n(11.92%) articles respectively.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 13:\nFurthermore, based on the ﬁndings in Table\n2, ANP\n(7.38%), PROMETHEE (6.62%), ELECTRE (8.65%), DEMATEL (1.78%) and VIKOR(3.56%) had next subsequent ranks.\nQuestion ﬁve was: (5) which journal published articles related to these MCDM\nmethods and approaches? Table\n12showed the results of this question, as we can see in\nthis table, from 120 journals; ranked ﬁrst was the European Journal of Operational\nResearch with 70 articles.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='bf8a48c3-62c4-4278-8890-fa8013c3336f', embedding=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='9a7c132f-2e5c-4fe3-9cbb-56892ea25e3a', node_type=None, metadata={'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='09e649f6a687513ba0eca7ad835446cd67846b4ffbd961096c05c84a00dd115d')}, hash='6789fd23ced91a72cd3aa7ac8636a22c88e209c434a2639d62fd5a5a7589a313', text='Source 14:\nAccording to this result, we can indicate that the Journal of\nExpert Systems andApplications and Applied Mathematical Modelling had the second\nand third rank with 20 and 13 articles respectively, although the Journal of Materials &\nDesign with 13 articles was ranked third. In other journal ranking the Journal of Civil\nEngineering and Management was ranked fourth with 12 articles. Moreover the Journal\nof Applied Mathematics and Computation , the Journal of Technological and Economic548 A. Mardani et al.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8355386131937158), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 15:\nreview article, to identify the scholarly articles related to DM methods in the database,\nwe have searched using several keywords, they were: MCDM, DM, AHP, COPRAS,TOPSIS, VIKOR, ELECTRE, MULTIMOORA, DEMATEL, SWARA, ANP, MOORA,PROMETHEE, WASPAS, SAW, FDM, ARAS, Entropy, Hybrid MCDM, and so on.\nAfter a preliminary search and collecting these scholarly articles, those relating to DM\ntechniques and approaches were selected.\n4. Results\n4.1.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 16:\n4. Results\n4.1. Classi ﬁcations and observations\nThis article is based on a literature review and classi ﬁcation of international journal arti-\ncles from 2000 to 2014. The majority of the journals are specialist journals in theMCDM area. For the purpose of this part of the article, some of journals are listedbased on publishers, and some journals (e.g. Web of Science database) are integratedbased on their publishers.\nResearch on MCDM continued and found many applications to different ﬁelds.\nMCDM provides strong DM in domains where selection of the best alternative is highlycomplex.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 17:\nMCDM provides strong DM in domains where selection of the best alternative is highlycomplex. This article reviews the main streams of considerations in MCDM theory and\npractice in detail, and we aimed to identify various applications and approaches and\nsuggest approaches that can be most robustly and effectively used to identify the bestalternative. This survey also addresses the problems in MCDM techniques. MCDMmethod has been applied to many domains to choose the best alternatives. Where manycriteria have come into existence, the best one can be obtained by analysing differentscopes of the criteria, weights of the criteria, and the selection of the optimum onesusing any MCDM techniques.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 18:\nThis article investigates the developments of various methods of MCDM and its\napplications. In our daily life, many decisions are made based on various criteria; thus\nthe decision can be made by assigning weights to different criteria and all the weightsare obtain from expert groups. It is important to determine the structure of the problemand explicitly evaluate multi-criteria. For example, in building a nuclear power plant,certain decisions have been taken based on different criteria.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 19:\nThere are not only verycomplex issues involving multi-criteria, some criteria may have an effect on some prob-lems; however, to have an optimum solution, all alternatives must have common cri-teria, which clearly lead to more informed and better decisions. The AHP method is\nused in the analysis of the health-safety and environmental risk assessment of re ﬁneries\nfor the location of the power plant, the risk factors such as health-safety risk, technologyrisk, etc. (Rezaian & Jozi,\n2012 ).\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 20:\n(Rezaian & Jozi,\n2012 ). TOPSIS has been applied to the selection of the best\nstrategic technology for the fuel cell in the automotive industry (Sadeghzadeh & Salehi,\n2011 ).\nIn all these articles, different methods have been used for different applications\nwhere each method has its own characteristics in ﬁnding the best alternatives.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313), NodeWithScore(node=TextNode(id_='4c0925ca-8f3c-448d-9dfc-847ebcb1db9c', embedding=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='09f22465-ec46-40fa-b5b8-d11a9a2beecd', node_type=None, metadata={'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, hash='1bc46955c4a79a03d4da934662266472fe2bebccdc855b49abdbd1887a8876e5')}, hash='8192e352c77cdeab12bf0a641d8ed9a60d98f020e2a4b7c80998f27833221509', text='Source 21:\nThe\napplications developed to solve multi-choice problems and the selected MCDM methods\nprovide better performance in cases such as Energy, environment and sustainability,\nSupply chain management, Material, Quality management, GIS, construction and projectmanagement, safety and risk management, manufacturing systems, technology manage-ment, operation research and soft computing, strategic management, knowledge manage-ment, production management, tourism management and other ﬁelds. Table\n1presents\nthe distribution of articles based on application ﬁelds.Economic Research-Ekonomska Istra živanja 521\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8338126026341313)], metadata={'b6bf0dce-a6a8-4697-aa32-2894a976d4f2': {'page_label': '90', 'file_name': 'Moreno-Jiménez et al_2020_Decision Support Systems X.pdf'}, 'bf8a48c3-62c4-4278-8890-fa8013c3336f': {'page_label': '34', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}, '4c0925ca-8f3c-448d-9dfc-847ebcb1db9c': {'page_label': '7', 'file_name': 'Mardani et al. - 2015 - Multiple criteria decision-making techniques and t.pdf'}})




```python
query("""DMDU uses the following steps: 1) Problem Framing
	- Identify
		- objectives
		- constraints
		- major uncertainties
		- definition of success.
2) Identify when the status quo starts to fail. 
	- Simulate Business as Usual
	- Identify tipping points into a failing system
		- Need to identify rules for switching interventions
3) Identify and measure interventions
4) Explore Pathways from interventions
5) Design Adaptive Plan


Provide a list of triplets that identifies the algorithms used with each part.\n\n\
                          """)

```

    This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (4096). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.



<b>

1. Problem Framing
	- Identify
		- objectives
		- constraints
		- major uncertainties
		- definition of success.
		(RDM)

2. Identify when the status quo starts to fail. 
	- Simulate Business as Usual
	- Identify tipping points into a failing system
		- Need to identify rules for switching interventions
		(RDM)

3. Identify and measure interventions
	(RDM)

4. Explore Pathways from interventions
	(RDM)

5. Design Adaptive Plan
	(RDM)

Note: The numbers in parentheses refer to the corresponding sources in the list of sources provided.</b>





    Response(response='\n\n1. Problem Framing\n\t- Identify\n\t\t- objectives\n\t\t- constraints\n\t\t- major uncertainties\n\t\t- definition of success.\n\t\t(RDM)\n\n2. Identify when the status quo starts to fail. \n\t- Simulate Business as Usual\n\t- Identify tipping points into a failing system\n\t\t- Need to identify rules for switching interventions\n\t\t(RDM)\n\n3. Identify and measure interventions\n\t(RDM)\n\n4. Explore Pathways from interventions\n\t(RDM)\n\n5. Design Adaptive Plan\n\t(RDM)\n\nNote: The numbers in parentheses refer to the corresponding sources in the list of sources provided.', source_nodes=[NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 1:\nbut  can  also  cover  uncertainties  related  to  the  data  or  models  that\nare  being  used  (Kwakkel  et  al.,  2010b ).\nThe  second  step  is  the  problem  analysis .  In  this  step,  the  current\nsituation  and  possible  future  situations  are  compared  to  the\nspeciﬁed  objectives  to  identify  whether  there  are  any  gaps.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 2:\nThe\npossible  future  situations  are  ‘reference  cases’  assuming  no  new\npolicies  are  implemented,  and  consist  of  (transient)  scenarios  that\nspan  the  uncertainties  identiﬁed  in  step  one.  A  gap  indicates  that\nactions  are  needed.  Both  opportunities  and  vulnerabilities  should\nbe  considered.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 3:\nBoth  opportunities  and  vulnerabilities  should\nbe  considered.  Opportunities  are  developments  that  can  help  in\nachieving  the  objectives,  while  vulnerabilities  are  developments\nthat  can  harm  the  extent  to  which  the  objectives  can  be  achieved.\nThe  identiﬁcation  of  opportunities  and  vulnerabilities  can  be  based\non  the  analysis  of  the  reference  cases,  which  can  best  be\naccomplished  using  a  computational  model.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 4:\nIn  the  third  step,  one  identiﬁes  possible  actions  that  can  be  taken\nto  meet  the  deﬁnition  for  success.  These  actions  can  thus  be\nspeciﬁed  in  light  of  the  opportunities  and  vulnerabilities  previ-\nously  identiﬁed  and  can  be  categorized  according  to  the  types  of\nactions  speciﬁed  in  the  Adaptive  Policymaking  framework  (i.e.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 5:\nshaping,  mitigating,  hedging,  and  capitalizing  actions).  The  aim  of\nthis  step  is  to  assemble  a  rich  set  of  possible  actions.  An\nidentiﬁcation  of  actions  for  different  perspectives  could  enforce\nthis  (e.g.  done  by  Offermans  et  al.,  2011 ).\nThe  fourth  step  is  to  evaluate  the  actions .\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 6:\nThe  fourth  step  is  to  evaluate  the  actions .  The  effects  of  the\nindividual  actions  on  the  outcome  indicators  are  assessed  for  each\nof  the  scenarios  and  can  be  presented  using  scorecards.  The  results\nare  used  to  identify  the  sell-by  date  for  each  of  the  actions.\nFurthermore,  the  vulnerabilities  and  opportunities  need  to  be\nreassessed.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 7:\nWas  the  action  able  to  reduce  or  remove  a  speciﬁed\nvulnerability?  Was  the  action  able  to  utilize  a  speciﬁed  opportu-\nnity?  Does  the  action  create  new  opportunities  and/or  vulner-\nabilities?\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 8:\nIneffective  actions  are  screened  out  (Walker,  1988 ),  and\nonly  the  promising  actions  are  used  in  the  next  steps  as  the  basic\nbuilding  blocks  for  the  assembly  of  Adaptation  Pathways.\nThe  ﬁfth  step  is  the  assembly  of  pathways  using  the  information\ngenerated  in  the  previous  steps.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='d3e2dd64-7e28-4968-a93e-bcf3eefae0bd', embedding=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='800df0d6-5f36-4d14-84e6-7d8160b694d4', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d7b58211ffef783a4bc0fd9738f01ce4c539827c2b1a088b421fbd4e71cb8742'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='728cac1f-4ee3-4d3d-b175-445b8c1893b1', node_type=None, metadata={'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='1b0308f51b9dcbe898cbb31a0f16aa2302c4a555d221f328236bbf8d9b10d6b9')}, hash='d9287c5dc70305a1e6515a2ff0cb676f083d7a8a9a64eb2fc2fdfc3000f816f7', text='Source 9:\nIt  is  conceivable  that  the\nreassessment  of  the  vulnerabilities  and  opportunities  in  the\nprevious  step  triggers  an  iterative  process  (back  to  step  3)  wherein\nnew  or  additional  actions  are  identiﬁed.  Once  the  set  of  actions  is\ndeemed  adequate,  pathways  can  be  designed.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8880535953587095), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 10:\n4.\nThe  ﬁrst  step  is  to  describe  the  study  area,  including  the  system’s\ncharacteristics,  the  objectives,  the  constraints  in  the  current\nsituation,  and  potential  constraints  in  future  situations.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 11:\nThe  result\nis  a  deﬁnition  of  success ,  which  is  a  speciﬁcation  of  the  desired\noutcomes  in  terms  of  indicators  and  targets  that  are  used  in\nsubsequent  steps  to  evaluate  the  performance  of  actions  and\npathways,  and  to  assess  the  ‘sell-by  dates’  of  the  actions.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 12:\nThe\ndescription  of  the  study  area  includes  a  speciﬁcation  of  the  major\nuncertainties  that  play  a  role  in  the  decisionmaking  problem.  These\nuncertainties  are  not  restricted  to  uncertainties  about  the  future,Table  1\nComparison  of  the  approaches.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 13:\nAspect  Adaptive  Policymaking  Adaptation  Pathways\nFocus  Starts  from  a  vision  of  the  decisionmaker  and  creates\na  plan  for  realizing  this  vision  and  protecting  it  from\nfailure.Explores  actions  for  achieving  objectives  over  time\nby  including  dynamic  interaction  between  the\nsystem  and  society.\nConsideration  of  the  multiplicity  of  futures  Indirectly  via  vulnerabilities  and  opportunities.  Explicitly  via  transient  scenarios.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 14:\nExplicitly  via  transient  scenarios.\nPlanning  process  Comprehensive  stepwise  approach  for  designing  a\nplan.Short  stepwise  approach  for  designing  Adaptation\nPathways.\nClarity  on  how  to  design  a  plan  Limited;  a  high  level  framework  that  can  be\ntranslated  into  a  speciﬁc  plan  in  many  different\nways.Application  oriented,  with  a  clear  link  to  the  use  of\nmodels  to  develop  a  speciﬁc  plan.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 15:\nTypes  of  actions  that  can  be  taken  Distinguishes  many  different  types  of  actions  that\ncan  be  taken  (e.g.  hedging,  mitigating,  and  shaping).No  speciﬁc  categorization  of  actions  is  used.  Several\nactions  and  pathways  are  presented.  A  variety  of\nactions  are  identiﬁed  based  on  different  societal\nperspectives.\nDesirable  plan  One  basic  plan  is  developed.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 16:\nDesirable  plan  One  basic  plan  is  developed.  No  clear  guideline  on\nhow  develop  the  basic  plan.Several  pathways  are  presented.  Different\nperspectives  result  in  different  preferred  pathways.\nNo  focus  on  how  to  identify  promising  pathways\nwhen  confronted  with  a  large  number  of  possible\nactions.\nConsideration  of  types  of  uncertainties  In  principle,  any  uncertainty  can  be  accounted  for.  In  principle,  any  uncertainty  can  be  accounted  for.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 17:\nIn  principle,  any  uncertainty  can  be  accounted  for.\nExplicit  attention  is  given  to  social  uncertainty.\nFlexibility  of  resulting  plan  Flexibility  is  established  through  the  monitoring\nsystem  and  associated  actions.The  Adaptation  Pathways  map  clearly  speciﬁes  when\na  policy  should  be  changed,  and  what  the  next  action\nshould  be.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='1fc10146-3df2-4254-931c-ca37073479f3', embedding=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='87b51d9f-9c9f-4a0e-88b6-b2789f908f0e', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='e99e122aa36fa2a7988cfa9906a604781b4afb846329b6d1376a0b68c339b48c'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='1084ffa4-3b16-47e1-b21f-bc47cf5ac189', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='d46075b6319371998ccc684ffa4af774a82d78619dbf9dfc49e64cfa5a7a7dc6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='6fb36557-3ee1-465d-bbd7-585eb0abaf64', node_type=None, metadata={'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, hash='07aaf42ecd074f727b51242b887f1d929ae28979de7daa2e90f4a1ab495fceaf')}, hash='9a80f257ad6c3a19d7defb09a1a43e0a55417e05191ee4e64ef7d976d2eae058', text='Source 18:\nDynamic  robustness  of  resulting  plan  Dynamic  robustness  results  from  the  monitoring  set\nup  in  Step  IV  and  the  actions  taken  in  Step  V.Dynamic  robustness  is  produced  indirectly  via  the\nidea  of  a  ‘sell-by  date’  and  the  shift  to  another  action.\nFig.  4.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8824753251334988), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 19:\nUncertainty (DMDU) is a subﬁeld of decision science\nthat focuses on developing and applying the frame-\nworks, tools and techniques necessary to produceactionable information while appropriately account-ing for deep uncertainty (Marchau et al. 2019 ). Mod-\nels have long been used to answer “what-if” questions(Bankes et al. 2013 ) and have been widely adopted in\nmany planning contexts, including in the ColoradoRiver Basin. DMDU builds on this model-informed\ndecision making by helping planners strategically\ndesign the “what ifs” and offering new quantitativetools to drive models and analyze output.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 20:\nSince theearly 2000s, a dedicated and rapidly growing commu-nity of researchers and practitioners forming theSociety for DMDU ( http://www.deepuncertainty.org )\nhas been applying and reﬁning DMDU methods in awide range of domains including national security\n(Dixon et al. 2008 ), energy planning (Toman et al.\n2008 ), and water resources management (Lempert\nand Groves 2010 ; Means et al. 2010 ; Basdekas 2014 ;\nRaucher and Raucher 2015 ; Groves et al. 2019 ).\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 21:\n2019 ).\nDMDU techniques share a common underlying\nphilosophy of designing iterative planning processesand analyses to identify actions that reduce a sys-tem’s vulnerability to uncertain future conditions.\nRobust Decision Making (RDM) (Lempert 2002 )i s\ndescribed here as an example because Reclamationhas used components of it in the past and is continu-ing to explore and develop related methods.RDM is designed to facilitate a “deliberation with\nanalysis” process through which parties can system-\natically integrate a large amount of information andwide-ranging positions about possible future condi-tions, important performance objectives, and appropri-ate actions to address challenges.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 22:\nThe ultimate goalsof RDM are to develop shared understanding of a sys-tem and identify a broadly acceptable plan or policythat is robust to a range of futures. The steps of RDM\nare depicted in Figure 3and summarized below. Note\nthat although the steps are numbered here anddescribed in order, in practice the relationshipsbetween the components of RDM are ﬂexible.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 23:\nIn Step 1, stakeholders deﬁne key components of\nthe analysis: performance objectives and criteria; thesets of possible actions, or strategies, that could beundertaken in pursuit of the objectives; ranges of val-\nues for uncertain future conditions; and a model that\nsimulates relationships between objectives, actions,and uncertainties. In Step 2, a proposed strategy istested in a wide range of future conditions to generatea thorough representation of performance variability.In Step 3, the performance information is analyzedwith statistical data mining techniques that seek to“discover” decision-relevant scenarios, or those combi-\nnations of uncertain conditions that cause the strategy\nto perform poorly.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 24:\n(This is a departure from traditionalanalysis in that the scenarios themselves are not dic-tated in advance of the procedure.) In Step 4, tradeoffs0%5%10%15%20%25%30%35%40%\n2021\n2022202320242025\n2026\n2027\n202820292030\n2031\n2032\n20332034\n203520362037\n2038\n20392040\n2041\n2042\n20432044\n2045\n2046\n2047\n2048\n2049\n2050Percent of Traces\nYearLake Powell < 3,\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 25:\n490 Feet in Any Month\n  CMIP3 Hydrology; 2016 Demands; Current Policies Con/g415nue\n  CMIP3 Hydrology; 2016 Demands; Policy Shi/g332s to No Ac/g415on Alterna/g415ve\n  CMIP5 Hydrology; 2016 Demands; Current Policies Con/g415nue\n  Full Hydrology; 2016 Demands; Current Policies Con/g415nue\n  Stress Test Hydrology; 2016 Demands; Current Policies Con/g415nue\n  Stress Test Hydrology; Alterna/g415ve Demands; Current Policies Con/g415nueScenario\nFIGURE 2.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116), NodeWithScore(node=TextNode(id_='4318cc2f-3150-409f-8370-f23c73f0b5df', embedding=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='bc36706c-cdc6-4bd3-9325-47fa198ea0eb', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='816516e2f53bec63bd79d1c7400b367f232c18e14c09e5e44a3f0b33cf041796'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e021fdea-4ff9-48b0-8f8e-da30662009c9', node_type=None, metadata={'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}, hash='22c9393c6bfd069392e6f468fb01f33aff92f7abc81660d40bc79132723de37e')}, hash='85c431c301565cb63fe7d7a2d625d0088eb6c03c5a5220bf3818e884f2ed6199', text='Source 26:\nCurrent Policies Con/g415nueScenario\nFIGURE 2. Risk projections for Lake Powell falling below 3,490 feet resulting from six future scenarios.\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.8793036793345116)], metadata={'d3e2dd64-7e28-4968-a93e-bcf3eefae0bd': {'page_label': '490', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, '1fc10146-3df2-4254-931c-ca37073479f3': {'page_label': '489', 'file_name': 'Haasnoot et al_2013_Dynamic adaptive policy pathways.pdf'}, '4318cc2f-3150-409f-8370-f23c73f0b5df': {'page_label': '5', 'file_name': 'Smith et al_2022_Decision Science Can Help Address the Challenges of Long-Term Planning in the.pdf'}})




```python

```