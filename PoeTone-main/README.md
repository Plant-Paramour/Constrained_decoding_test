# PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs

This repository contains the codebase for the paper: **PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs**


## Environment and Dependencies

To install all required dependencies, run:
```bash
pip install -r requirements.txt
```

## Code and Database
For evaluating LLMs' capability of generating Chinese Songci, run the python files under evaluation/  
llama3.py  
mistral.py  
deepseekr1.py  
qwen3.py  
gpt4o.py  

For evaluation  
evaluation_all  

For cleaning the generated Songci  
raw_data_cleaning.py  

For applying the generate-critic method to fine-tune the LLMs  
create_best_of_n_dataset.py  
run_sft.py  
