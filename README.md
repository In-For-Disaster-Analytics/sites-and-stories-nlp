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

## Running the Coookbook

### Parameters

- **Update Cookbook:** This parameter is used to update the notebooks of the cookbook to the latest version.

- **Update Conda Environment:** This parameter is used to update the conda environment to the latest version.

Note: Both parameters are not applicable on your first run. These options are designed for enhancing subsequent uses.

### Initial Setup

The initial run will install the necessary packages and download the required models. Please be prepared for the initial run to take approximately 15 minutes.

### Queues

Lonestar6 hosts [84 A100 GPU nodes](https://docs.tacc.utexas.edu/hpc/lonestar6/#table2) that are configured identically to the compute nodes with the addition of 3 NVIDIA A100 GPUs. Each A100 GPU has a peak performance of 9.7 TFlops in double precision and 312 TFlops in FP16 precision using the Tensor Cores.

For more information, please see the [Lonestar6 User Guide](https://docs.tacc.utexas.edu/hpc/lonestar6/#table5).

### Allocation

This cookbook requires an allocation on TACC systems. If you do not have an allocation, you can request one [here](https://portal.tacc.utexas.edu/allocations).
