# dynamic-tstorm-opm
Model and evaluation code for the dynamic outage prediction models described in "Dynamic Modeling of Power Outages Caused by Thunderstorms".

First, preprocess the dataset described in the manuscript (and included here), performing scaling and PCA on a leave-one-storm-out (LOSO) cross-validation basis:
```{bash}
mkdir data/LOSO
python3 generate_PCA_features.py
```

To produce LOSO cross-validated predictions from the LSTM, run
```{bash}
python3 lstm_CT_loso.py
```

To perform the hyperparameter searches and produce predictions from the Poisson regression, KNN, and random forest models, run
```{bash}
Rscript dynamic_OPMs_search_predict.R
```
