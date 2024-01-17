

## Motivation

The Sites and Stories cookbooks are to provide an environment to run Large Language Models (LLM) on TACC systems. Specifically, this cookbook shows how to use LLM's to generate a Retrieval-Augmented Generation (RAG) query a personalized pdf library as a corpus. 

This environment uses the [LLAMA Index](https://www.llamaindex.ai/) python library to organize and query the corpus, and downloads/uses [Hugging Face](https://huggingface.co/models?other=text-generation-inference) models for the queries. 

## Authors
William Mobley - wmobley@tacc.utexas.edu
## Contributors
<a href="https://github.com/In-For-Disaster-Analytics/sites-and-stories-nlp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=In-For-Disaster-Analytics/sites-and-stories-nlp" />
</a>

## Structure

## Foundations
There are currently two files to run the RAG model. The [LLamaIndex.ipynb](LLamaIndex.ipynb) file streamlines access to the LLMs; loads the requisite corpus and allows the user to query the pdf's a retrieves answers with citations. 

The [LLM_location.py](LLM_location.py) file provides the framework to run the notebook. In this file you can add other LLM Models and provide further customization. 

Alternatively you can use pieces within the LLM_location file to develop your own system. 

## Running the Notebooks
