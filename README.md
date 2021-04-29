# bertron
(toy) BERT authorship assignment against Enron Corpus

------------
### Some small inspirations  
- http://publications.idiap.ch/downloads/papers/2020/Fabien_ICON2020_2020.pdf
- https://link.springer.com/content/pdf/10.1007%2F978-3-540-30115-8_22.pdf

BERT multiclass classsification:
- https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
  - From: https://www.coursera.org/projects/sentiment-analysis-bert

------------------
### Instructions
Download Enron Corpus here: https://www.cs.cmu.edu/~enron/
 - Save as `data/maildir` for data formatting to work  

Then run `bertron/format_data.py` to generate processed data sets  
Then run `bertron/train_bert.py > bertron/models/model_output.txt` to see some results!
- n.b.: still a work in progress

------------------
### Environment
Running on EC2 Deep Learning AMI instance:  
- `c5.12xlarge`, "advanced CPU compute-intensive workloads" (only because they wouldn't requisition any GPUs to me: `g3.4xlarge`)  
- PyTorch 1.7.1 with Python3.7 (CUDA 11.1 and Intel MKL)  
- https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html  

------------------
### Results
Model performance and output:
- Metrics by epoch (F1 score): `/bertron/models/model_output.txt`
  - stopped after 2 (of 4) epochs (on ~69k records), trained for ~12 hours
- sample predictions: `/notebooks/predict`
  - a few sample predictions at the end of the notebook (using epoch 1 model checkpoint)

------------------
### Future work:
- Additional performance metrics: AUC ROC, etc/
- More intelligent tokenizer treatment (currently using default values)
- More exhaustive search for best pre-trained model starting point (currently using https://huggingface.co/bert-base-uncased)
  - Case may be very important
  - Find most relevant starting corpus ([BooksCorpus and English Wikipedia](https://arxiv.org/pdf/1810.04805.pdf) were used for pretrained model, may not best reflect language patterns of emails)
- Better document preparation
  - consider removing any "signatures" which are obviously extremely identifiable
- Baseline (naive) model
  - Something like tf-idf with random forest to evaluate incremental benefit of complex BERT-based model
- The list goes on...
