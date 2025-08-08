# telemed_agent
# How to train and test model:
- Step 1.1: If using linux: Create venv
- Step 1.2: If using windows, create wsl before create venv to ensure the cuda can train on GPU
- Step 2: Run terminal 'pip install -r Requirements.txt'
- Step 3: Download the pdf doc for RAG in ./data/RAG/
- Step 4: Run fine_tune, run rag and save the models
- Step 5: Run terminal 'python3 main.py'
# How to test web:
- Step 1: Run terminal 'pip install streamlit'
- Step 2: Run terminal 'streamlit run app.py'
Notice: if you change the architecture of the folder, please check all the path variable before running
