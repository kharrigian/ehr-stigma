# Sample Jobs

Not all experiments presented in our ACL paper can be reproduced due to the private nature of our institution's dataset. However, all experiments that solely use the MIMIC-IV dataset can be reproduced using the code in this repository. The list below explains which scripts are relevant for each set of experiments. We also include additional utilities which may be useful for some end users.

1. `run_train_baseline.sh` - Used to train the Majority Overall, Majority Per Anchor, LR (Context), and LR (Context + Anchor) models in Section 4.1.
2. `run_train_bert.sh` - Used to train all Base BERT and Clinical BERT models in Section 4.1. 
3. `run_train_comparison.sh` - Used to aggregate model performance numbers from the prior two scripts.
4. `run_domain_distance.sh` - Used for experiments in Section 4.2. Attempt to infer demographic attributes from anchor embeddings.
5. `run_apply_baseline.sh` - Provides option to apply the non-BERT models to a new sample of notes. Assumes the notes presented as input have already had the appropriate keywords extracted.
6. `run_apply_bert.sh` - Same as (5), but for the BERT models.