# bertron
(toy) BERT authorship assignment against Enron Corpus

Some small inspirations  
- http://publications.idiap.ch/downloads/papers/2020/Fabien_ICON2020_2020.pdf
- https://link.springer.com/content/pdf/10.1007%2F978-3-540-30115-8_22.pdf

BERT multiclass classsification:
- https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
  - From: https://www.coursera.org/projects/sentiment-analysis-bert

Download Enron Corpus here: https://www.cs.cmu.edu/~enron/

Save as `data/maildir` for data formatting to work  
Then run `bertron/format_data.py` to generate processed data sets  
Then run `bertron/train_bert.py > bertron/models/model_output.txt` to see some results!