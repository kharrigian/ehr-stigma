# Model Training Configuration Files

The JSON files hosted in this directory were used to train the models presented in our ACL 2023 paper.

* `baseline-majority.default.json`: Majority Overall
* `baseline-statistical.default.json`: Majority Per Keyword, Context Only, Keyword + Context
* `bert-full.default.json`: Anchor Mean pooling for the Clinical BERT and Base BERT models.
* `bert-full.cls.json`: CLS token classification for the Clinical BERT and Base BERT models
* `bert-full.pooler.json`: BERT Pooler classification layer for the Clinical BERT and Base BERT models.
* `bert-full.sentence-mean.json`: Mean pooling of the tokens in the context window for the Clinical BERT and Base BERT models.
* `bert-classifier_only.default.json`: Freezes the BERT encoders, uses Anchor Mean pooling. *Not* used in the ACL 2023 paper.