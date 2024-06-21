## Todo:
- Setup ollama with Ipywidgets for each model
    - Default to Lightweight Model
- Consistent:
	- Styling of Tutorials
	- Naming Conventions
       - See High Level Name
	- Folder Structure
         - Basic Prompting
         - Data Management
         - Advanced LLM Systems
         - Evaluation
- Incorporate ollama into RAG
- Switch to Lang Chain?
## Current Tutorials
0) What is a LLM introduction to LLMs
1) 
### Topic Analysis
1)  Quickstart Berttopic
2)  
### Basic Prompting
2) Prompting
	- Current Name: Prompt_Engineering_OpenAI
	- Description: Basic Prompting for a chat bot. No additional information than the model
	- Current Setup: F1 Racing
	- Works Ollama: True
	- Dataset: False
3) Vertical_Prompting
	- Current Name: Vertical Chat
	- Current Setup: Order bot
	- Description: Code that allows for "memory" in the model. Currently uses synthetic menu. 
	- Works Ollama: True
	- Dataset: False
### Data management
4) Vector_Databases
	- Current Name: how-to-use-a-embedding-database-with-a-llm-from-hf
	- Description: Saving and querying a chromadb vector database. Currently uses Newspapers in a csv for its query. 
	- Current Setup: NewsPaper Query
	- Works Ollama: True
	- Dataset: labelled_newscatcher_dataset.csv

### Advanced LLM systems
5) Retrieval_Augmented_Generation
	- Current Name: 3_4_Medical_Assistant_Agent
	- Current Setup: Medical Data Query
	- Works Ollama: True
	- Dataset: MedQuad-MedicalQnADataset
6) Topic Analysis and LLMs
7) RAG based on Journal Articles to Knowledge Graph

#### Can Stop Here for First Tutorials
6) Moderation System
	- Name: HF_gpt_j_6b_Moderation_System
	- Description: This is basically a sentiment analysis. Not sure its needed. 
	- Current Setup: Synthetic Data
	- Works Ollama: False
	- Dataset: False
7) Data Analyst Agent
	- Name: LangChain_Agent_create_Data_Scientist_Assistant
	- Description: Basic Data analysis of an excel file. This one is pretty cool. May not be necessary for sites and stories but likely useful  
	- Current Setup: Climate Data
	- Works Ollama: False
	- Dataset: climate-insights-dataset
#### Do these.
### Evaluation
8) ROUGE Metrics
	- Name: rouge-evaluation-untrained-vs-trained-llm
	- Description: How to evaluate Summarization based on the ROUGE metric
	- Current Setup: News Articles
	- Works Ollama: False
	- Dataset: /mit-ai-news-published-till-2023/articles.csv
9) Evaluating Summarizations with LangSmith using Embedding Distance
	- Name: LangSmithSumarizations
	- Description: Continuation of Rouge Metrics
	- Current Setup: Daily Mail
	- Works Ollama: False
	- Dataset: ccdv/cnn_dailymail
10) langsmith_Medical_Assistant_Agent
	- Name: LangChain_Agent_create_Data_Scientist_Assistant
	- Description: Example of a complex rag and how the model is tracing through the system. Good For debugging
	- Current Setup: Medical Data Query
	- Works Ollama: False

### Fine Tuning
11) Introduction to LoRA Tuning using PEFT from Hugging Face
	- Name: LoRA_Tuning_PEFT
	- Description: LoRA is a re-parameterization technique. Its operation is simple, complex, and brilliant at the same time. It involves reducing the size of the matrices to be trained by dividing them in such a way that when multiplied, they yield the original matrix.
	- Current Setup: fka/awesome-chatgpt-prompts
	- Works Ollama: False
12) Prompt Tuning
	- Name: Prompt_Tuning_PEFT
	- Description: It’s an Additive Fine-Tuning technique for models. This means that we WILL NOT MODIFY ANY WEIGHTS OF THE ORIGINAL MODEL. You might be wondering, how are we going to perform fine-tuning then? Well, we will train additional layers that are added to the model. That’s why it’s called an Additive technique.
	- Current Setup: Medical Data Query
	- Works Ollama: False