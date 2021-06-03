<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            displayMath: [['$$','$$']],
            inlineMath: [['$','$']],
        },
    });
</script>

# Project Report: Flavors of Physics, The Strange D Meson Decay
## Problem Definition
### Domain Background


This project is a particle physics problem. Its name is inspired by what physicists call "[flavor](https://en.wikipedia.org/wiki/Flavour_(particle_physics))", the species of an elementary particle. The  [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) of particle physics is a well-established theory that explains the properties of fundamental particles and their interactions, describing the "flavor" of each particle. As mentioned in Charlotte Louise Mary Wallace CERN [Thesis](https://cds.cern.ch/record/2196092/files/CERN-THESIS-2016-064.pdf), the Standard Model theory has been tested by multiple experiments, but despite its successes, it is still incomplete, and further research is needed. 

The Standard Model counts six flavors of quarks and six flavors of leptons, as shown below. "Flavor" is essentially a [quantum number](https://en.wikipedia.org/wiki/Flavour_(particle_physics)#Quantum_numbers) that characterizes the quantum state of those quarks.

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Standard_Model_of_Elementary_Particles.svg/1024px-Standard_Model_of_Elementary_Particles.svg.png" alt="drawing" width="300"/>
</div>

 The Ds decay project is influenced by a CERN [kaggle competition problem](https://www.kaggle.com/c/flavours-of-physics/overview/description) about the flavors of physics. In the initial problem, scientists try to find if it is possible the τ (*tau*) lepton to [decay](https://en.wikipedia.org/wiki/Particle_decay) (transform into multiple other particles) to three μ (muon) leptons. The problem I chose, however, concerns the [Ds meson](https://en.wikipedia.org/wiki/D_meson) or *strange D meson*, a composite particle that consists of one quark or one antiquark, and how often it decays into a *φ* ([phi meson](https://en.wikipedia.org/wiki/Phi_meson)) and a *π* ([pi meson or pion](https://en.wikipedia.org/wiki/Pion)) based on multiple factors. The decay is described by the following flow:

$$D_s \to φπ$$

You can see where the meson belongs in the subatomic particles map below. The purple part describes the composite particles.

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Particle_overview.svg/1920px-Particle_overview.svg.png" alt="drawing" width="400"/>
</div>

Ander Ryd in his [paper](https://wiki.classe.cornell.edu/pub/People/AndersRyd/DHadRMP.pdf) argues that the D meson decays have been a challenge, though scientists have been focused on their decays since the particle discovery. As a result, the existing dataset of this project is sufficient and based on well-studied experiment observations.


### Problem Statement

The problem falls into the category of binary classification problems. Based on particle collision events (that cause the $D_s \to φπ$ decays) and their properties, I am challenged to train a machine learning model that predicts whether the decay we are interested in happens in a collision. The model will be trained in the train set of the existing dataset, and it will be evaluated on the testing set.

## TODO: Data Analysis

As described in the [flavors of physics](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test) project, the $D_s \to φπ$ decay has a very similar topology as the *tau* decay, and their datasets share almost the same features. In the *tau* decay problem, the Ds decay data is used as part of the CERN evaluation process. This dataset will be used as the main dataset of the $D_s \to φπ$ decay problem solution.

This is a labeled dataset of 16.410 samples and 46 features, which are described below. The *signal* column classifies the samples into *signal events* (decays happening) denoted with *1* and *background events* (decays not happening) denoted with *0*. The dataset is balanced, with 8.205 signal events and 8.205 background events. 

The features of the dataset are described below:
* FlightDistance - Distance between Ds and PV (primary vertex, the original protons collision point).
* FlightDistanceError - Error on FlightDistance.
* LifeTime - Life time of Ds candidate.
* IP - Impact Parameter of Ds candidate.
* IPSig - Significance of Impact Parameter.
* VertexChi2 - χ2 of Ds vertex.
* dira - Cosine of the angle between the Ds momentum and line between PV and *Ds* vertex. 
* pt - transverse momentum of Ds.
* DOCAone - Distance of Closest Approach between p0 and p1.
* DOCAtwo - Distance of Closest Approach between p1 and p2.
* DOCAthree - Distance of Closest Approach between p0 and p2.
* IP_p0p2 - Impact parameter of the p0 and p2 pair.
* IP_p1p2 - Impact parameter of the p1 and p2 pair.
* isolationa - Track isolation variable.
* isolationb - Track isolation variable.
* isolationc - Track isolation variable.
* isolationd - Track isolation variable.
* isolatione - Track isolation variable.
* isolationf - Track isolation variable.
* iso - Track isolation variable.
* CDF1 - Cone isolation variable.
* CDF2 - Cone isolation variable.
* CDF3 - Cone isolation variable.
* ISO_SumBDT - Track isolation variable.
* p0_IsoBDT - Track isolation variable.
* p1_IsoBDT - Track isolation variable.
* p2_IsoBDT - Track isolation variable.
* p0_track_Chi2Dof - Quality of p0 muon track.
* p1_track_Chi2Dof - Quality of p1 muon track.
* p2_track_Chi2Dof - Quality of p2 muon track.
* p0_pt - Transverse momentum of p0 muon.
* p0_p - Momentum of p0 muon.
* p0_eta - Pseudorapidity of p0 muon.
* p0_IP - Impact parameter of p0 muon.
* p0_IPSig - Impact Parameter Significance of p0 muon.
* p1_pt - Transverse momentum of p1 muon.
* p1_p - Momentum of p1 muon.
* p1_eta - Pseudorapidity of p1 muon.
* p1_IP - Impact parameter of p1 muon.
* p1_IPSig - Impact Parameter Significance of p1 muon.
* p2_pt - Transverse momentum of p2 muon.
* p2_p - Momentum of p2 muon.
* p2_eta - Pseudorapidity of p2 muon.
* p2_IP - Impact parameter of p2 muon.
* p2_IPSig - Impact Parameter Significance of p2 muon.
* SPDhits - Number of hits in the SPD detector.
* signal - This is the target variable.

### Obtain the dataset
There are three ways to get the data described above:
* I recommend downloading the resampled dataset from the Github repo I created for this project. I intend to use this resampled dataset, as the original is heavily imbalanced. The resampled dataset is also smaller and easier to manage, which makes it more suitable for this Udacity project. I made sure that the dataset has sufficient data for my analysis. If you want to get the original dataset, follow the next point.
* From Kaggle, by downloading the *check_agreement.csv.zip* from [here](https://www.kaggle.com/c/flavours-of-physics/data?select=check_agreement.csv.zip) (this requires a Kaggle account).

> Note that in the resampled dataset, I have dropped the "weights" feature from the original dataset as, according to the [description of the dataset](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test), is the feature used to determine if the decay happens or not based on its value, and is the one used to create the binary *signal* column. It will not be used in the solution whatsoever.

TODO: Add the distribution of data and say they need scaling!!

## Algorithms Implementation

This is a binary classification problem, so the solution will be the output of a binary classifier. There is no constrain in using any classifier in particular for this problem. However, in the [evaluation description](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test) of the Kaggle competition described so far, is it mentioned that the [Kolmogorov–Smirnov (KS)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) test is used to evaluate the differences between the classifier distribution and the true *signal* values distribution. Also, the KS test should be less than 0.09.

### Models Exploration
To solve this problem, I trained a number of binary classifiers, using different hyperparameters tuning methods, in combination with different data scaling methods. For all the different results, a number of performance metrics are gathered in a single table, and the best model is chosen based on the metrics values. The metrics are described in more detail in the *Models Performance* section. As part of the project proposal, I trained a benchmark model presented below.


### Benchmark Model

TODO: add details about how the kNN works

As a benchmark model, I use a simple k-Nearest Neighbor classifier, and grid search for tuning the k hyperparameter. The benchmark model script can be found [here](https://github.com/Punchyou/flavors_of_physics_Ds_decay/blob/master/knn_benchmark_model.py). The execution of that script generates the following plot. The plot shows the accuracy of the kNN model for each one of the k values (from 1 to 80). The best model is the kNN model with k=52, and highest accuracy of 71%.

<div align="center">
<img src="https://raw.githubusercontent.com/Punchyou/flavors_of_physics_Ds_decay/master/images/knn_benchmark_acc.png" alt="drawing" width="400"/>
</div>

### Improving the Benchmark Model
To improve on the benchmark model, I trained a number of different binary classifiers and compared their performance. I have also scaled the data in different ways. Scaling of the input data is a requirement for most machine learning estimators in this project, as the data might behave badly when individual features do not are not normally distributed (see the distribution of the features plots above). An example is a Support Vector Machines model (presented below) which assumes that all features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

All scaled methods were used in combination with all the models. Also, different methods of hyperparameter tuning were used for each model.

##### Scaling methods used
* `sklearn`'s [Standard Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): 

$$z = (x - u) / s$$

where $u$ is the mean of the training samples or zero if `mean=False`, and $s$ is the standard deviation of the training samples or one if `std=False`.

* `sklearn`'s [Minmax Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html):

$$X_{scaled} = X_{std} * (max - min) + min$$
where min, max is the features range

This estimator scales and translates each feature individually such that it is in the given range on the training, zero and one in this case.

* `sklearn`'s [Robust Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html):

Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range.
ß
##### Binary Classifiers trained
The performance metrics are presented and compared at the end of this section for all the models, and all the different scaling methods.

The models used are:
* kNN, tuned with the Grid Search Method
The details of the kNN model are explained in the benchmark model section. For this model the [Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) method for hyperparameter tuning was used, with the ranges of parameters from 0 to 60.

STOPPED HERE

* Support Vector Machines, tuned with the Random Search
* Stochastic Gradient Descent, tuned
* XGBoost with Bayes optimization
* XGBoost with Bayes optimization - second version
 


## Models Performance

The evaluation metric used to choose the k values of the kNN benchmark model is the `sklearn` accuracy. It is calculated by dividing the number of correct predictions by the total number of samples. As the dataset is balanced with equal class distribution, the [accuracy paradox](https://en.wikipedia.org/wiki/Accuracy_paradox) is avoided and the metric does not provide missleading information about the models performance.


5. **Model Evaluation**: A number of performance metrics will be used for this step. Evaluation metrics suitable for this problem might be [false positive rate, false negative rate, true negative rate, negative predictive value](https://neptune.ai/blog/evaluation-metrics-binary-classification), accuracy or Kolmogorov–Smirnov test.

A workflow that sums up the steps above is showm below. This workflow is part of a typical data science lifecycle, as presented [here](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/lifecycle).

<div align="center">
<img src="https://raw.githubusercontent.com/Punchyou/flavors_of_physics_Ds_decay/master/images/project_ds_workflow.png" alt="drawing" width="350"/>
</div>

## Conclusion


## Sources:
* https://en.wikipedia.org/wiki/Flavour_(particle_physics)
* https://cds.cern.ch/record/2713513?ln=en
* https://en.wikipedia.org/wiki/Standard_Model
* https://cds.cern.ch/record/2196092/files/CERN-THESIS-2016-064.pdf
* https://en.wikipedia.org/wiki/Particle_decay
* https://en.wikipedia.org/wiki/D_meson
* https://en.wikipedia.org/wiki/Phi_meson
* https://en.wikipedia.org/wiki/Pion
* https://wiki.classe.cornell.edu/pub/People/AndersRyd/DHadRMP.pdf
* https://www.kaggle.com/c/flavours-of-physics/overview
* https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
* https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/lifecycle
* http://cds.cern.ch/record/2668282/files/BPH-17-004-pas.pdf
* https://neptune.ai/blog/evaluation-metrics-binary-classification
* https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/
* https://en.wikipedia.org/wiki/Accuracy_paradox
* http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
* https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

Dataset Issues:
    * High acciracy due to the dataset
    * Pickec another dataset, but it was imbalanced
    * Had to make it balances by resampling the dataset


How to choose the best n components for pca:
* PCA project the data to less dimensions, so we need to scale the data
beforehand.

* A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. This can be determined by looking at the cumulative explained variance ratio as a function of the number of components
This curve quantifies how much of the total, 64-dimensional variance is contained within the first N components.

* From the pca graph, looks like that the data are described from 3 variables.

Why use sklearn pipeline for SGDClassifier?
Often in ML tasks you need to perform sequence of different transformations (find set of features, generate new features, select only some good features) of raw dataset before applying final estimator. Pipeline gives you a single interface for all 3 steps of transformation and resulting estimator. It encapsulates transformers and predictors inside.


From he hist() plot of the training data:
If we ignore the clutter of the plots and focus on the histograms themselves, we can see that many variables have a skewed distribution.

The dataset provides a good candidate for using a robust scaler transform to standardize the data in the presence of skewed distributions and outliers.

How to do the capstonre project report: https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_report_template.md

TODO: The dataset has been resampled, due to the amount of the first dataset, only the resampled dataset is present in the project. The code used for resampling is the following:
```py
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

. The problem with accuracy this metric is that when problems are imbalanced it is easy to get a high accuracy score by simply classifying all observations as the majority class


### learning curves
"""The first plot is the learning curve
The plots in the second row show the times required by the models to train with various sizes of training dataset. The plots in the third row show how much time was required to train the models for each training sizes."""


"""The first plot is the learning curve
The plots in the second row show the times required by the models to train with various sizes of training dataset. The plots in the third row show how much time was required to train the models for each training sizes."""


TODO: There are not nan values in the dataset.
TODO: merge o,ages amd plots folder  IN **IMAGES**
TODO: add feature importance

Sources:
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
Data Exploration - Model selection:
https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
Data Science from Scratch
Machine learning with python
Feature Egnineering
https://github.com/Punchyou/blog-binary-classification-metrics
https://www.youtube.com/watch?v=aXpsCyXXMJE
Models Evaluation:
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    * https://neptune.ai/blog/evaluation-metrics-binary-classification
Hyperparameters Tuning:
    * https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
    * https://towardsdatascience.com/a-guide-to-svm-parameter-tuning-8bfe6b8a452c
Model Selection: https://machinelearningmastery.com/types-of-classification-in-machine-learning/
https://www.kaggle.com/nanomathias/bayesian-optimization-of-xgboost-lb-0-9769

Scaling
https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/