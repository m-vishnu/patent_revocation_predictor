# IP Analytics task solution

Hi there! Thanks for taking the time to read this.

- pip install -r requirements.txt should install all the packages you need
- Please take a look at the 'notebooks/eda.ipynb' notebook to see some charts and my thought process behind creating them
- The 'notebooks/lda_exploration' folder  has some failed experiments trying to create topic models for the text data
- You may need to run nltk_download.ipynb just to make sure all the stopwords, etc are downloaded


### The models:

- The create_tfidf_dataset creates the data frame required as input for hte model fitting stage
- The model fitting is a pipeline that combines gridsearchCV and a feature selection step. 
    - src.model_selection_* has scripts that creates the model selection pipeline.
    - Of all the tasks remaining, cleaning up the code in these files is top priority :(

- We investigated 2 ways to build the models: 
    - Translate all text to english using GPT-4o-mini so that we dont lose info from DE/FR rows
    - TF-IDF transform the text from title+summary to fit the models
    - Use openAI embeddings (generated using text-embedding-3-small) for each row. This 1536 dimensional embedding is projected down to ~250 dimensions using PCA, preserving 90% of the variance
    - The 'notebooks/final_model_performance/model_performance_analysis.ipynb' has the final model performance analysis, confusion matrix ,etc.
  