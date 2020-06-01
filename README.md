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

To produce the LOSO cross-validated hyperparameter searches and predictions from the Poisson regression, KNN, and random forest models, run
```{bash}
Rscript dynamic_OPMs_search_predict.R
```
## Software references
* F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
* M. Abadi et al., “Tensorflow: A system for large-scale machine learning,” in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pp. 265–283, 2016.
* F. Chollet et al., “Keras.” https://keras.io (accessed May 21, 2020), 2015.
* S. van der Walt et al., “The numpy array:  A structure for efficient numerical computation,” Computing in Science Engineering, vol. 13, no. 2, pp. 22–30, 2011.
* W. McKinney, “Data Structures for Statistical Computing in Python,” in Proceedings of the 9th Python in Science Conference, pp. 56–61, 2010.
* J. D. Hunter, “Matplotlib: A 2d graphics environment,” Computing in Science Engineering, vol. 9, no. 3, pp. 90–95, 2007.
* M. Kuhn, “Building predictive models in r using the caret package,” Journal of Statistical Software, Articles, vol. 28, no. 5, pp. 1–26, 2008.
* M. Wright and A. Ziegler, “ranger: A fast implementation of random forests for  high  dimensional  data  in  c++  and  r,” Journal of Statistical Software, Articles, vol. 77, no. 1, pp. 1–17, 2017.
