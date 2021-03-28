# PracticumPhysics
The flavors of physics kaggle competition completed for Regis University's MSDS Practicum 1

# Explanation of Work
My presentation is Practicum 1.pdf that explains my work


# Competition Website
https://www.kaggle.com/c/flavours-of-physics
Please download the following files from the kaggle competition to perform this model:
- check_agreement.csv
- check_correlation.csv
- sample_submission.csv
- test.csv
- training.csv

# The Final Model
### The file "FinalModels" ends with my final two models for my prediction.
GBC1.pkl and GBC2.pkl are my saved models.
I weighted the predictions 27% GBC1 and 73% GBC2

# Data
The training data is in training.csv
There are 3 test data sets: test.csv, check_correlation.csv, and check_agreement.csv

# Data Exploration
I performed all my data exploration in R and can be found in the EDA.R file.

# Data Cleaning
There appeared to be outliers in the data when looking at boxplots, however the same outliers are also
seen in the test set and they could be important clues to finding this unknown science.
I cleaned the training file after splitting it into a train (tn) and validation (val) sets because
validation is only performed on data points with an ANN value greater than 0.4  so I romoved any lower
and put them back into the training set to be a 20/80 split.

# Tuning Hyperparameters
The hyperparameter files are the different models I tested to tune my model

# Submission
- The JH1101620 was my final submission file to kaggle
- 181008 was my previous prediction on my Tuesday 10/16/2018 presentation making the JH101618 prediction.


# Problem Statement

The purpose of this project is to identify Lepton Flavor Violations (LFVs).  A LFV is a transition in the [lepton space](https://en.wikipedia.org/wiki/Subatomic_particle#/media/File:Standard_Model_of_Elementary_Particles.svg) that does not conserve the Lepton number (a quantum property of lepton types).  This is significant because such a change would indicate a _violation_ of the Standard Model of physics.  The data is from the LHCb (Large Hadron Collider-beauty) experiment at CERN.  Being able to detect such events using machine learning is critical because of what they suggest and because a ML method can process exponentially more data and much faster than a scientist could.

-------

# Executive Summary

This data is from the [Flavors of Physics](https://www.kaggle.com/c/flavours-of-physics-kernels-only/overview) Kaggle Competition.

As a group, we are are interested in physics and especially the exotic objects, especially the various fields of quantum dynamics and the cutting-edge experimentation being done at CERN.  When we saw this, we knew we had to attempt it.

The data were assembled from a database so it was very clean: no missing data or anything looked wrong to us.  The only real cleaning we did was to shorten or reformat the column names.  The data was difficult when visualizing because there are roughly 50 features in the set including the target.  Apart from the target, which is binary, all of the features contain numeric data.  The large number of features made it difficult to satisfactorily visualize the data because we would have ended up with more than 100 charts, which is not ideal: we chose to use violin plots because they visualize the distribution of the data over a box plot.  When visualizing the data we noted that the target is the majority of the data, which we have not encountered before and is unusual because the description of the data suggested this phenomenon is rare.  This is a concern because it suggests that the data may have been altered to have a good balance of classes.

When we were planning our feature engineering, we realized that they are no exceedingly strong correlations amongst features so we changed our focus to reduction of dimensionality.  Since there are so many features, we figured that exposing the models to all features in the data would significantly increase the variance in the models.  To that end we decided to use this project to compare the performance of models with data subject to Principal Component Analysis (PCA) and the raw data.

We chose to use a total of five classification models composed of three types: distance-based, tree-based, and boosted logistic regression models.  Each of the five models algorithms produced two models: one based on the original data and one based on the data put through PCA.  Comparing models was based on two sets of evaluators: a confusion matrix and a set of four classification metrics.  The best model was chosen on a combination of the two but also by comparing the PCA and original data.  Once we had chosen our best model, we plotted a ROC curve and visualized the metric scores.

-------

# Conclusions & Recommendations

After tuning and evaluating each model, we were able to determine that our best model is an XGBoost classifier with the original data.  This was surprising to us because we had believed that PCA would improve our models' performance.

It was difficult choosing our best model because, while our best model had better test scores, our second best model was much less overfit.  Due to the potential significance of the results, we felt it was best for us to choose our highest performing model.  Overall our model scores were very high, especially our AUROC and Matthews Correlation Coefficient , 0.88 and 0.76, both of which indicate a model with high performance.  Additionally, we had minimal numbers of false positives and negatives.

-------

# Next Steps

While we are confident with our model's performance, we believe there is still room for significant improvement.  Going forward, we would like to experiment with feature engineering and further tune the model.  That being said, time is a limiting factor: the XGBoost algorithm is not the fastest and can take a very long time when grid-searching.

Further down the road, we are considering creating a feed-forward neural network to see how one would perform in comparison.

------------------------ FROM Kaggle ----------------------------
# The problem

This idea is taken from ecisting kaggle competition: [Flavors of physics](https://www.kaggle.com/c/flavours-of-physics/overview). The goal of the completition is to find a phenomenon that is not already known to exist – charged lepton flavour violation – thereby helping to establish "[new physics](https://en.wikipedia.org/wiki/Physics_beyond_the_Standard_Model)".

## Background - Flavours of Physics 101

The laws of nature ensure that some physical quantities, such as energy or momentum, are conserved. From [Noether’s theorem](https://en.wikipedia.org/wiki/Noether%27s_theorem), we know that each conservation law is associated with a fundamental symmetry. For example, conservation of energy is due to the time-invariance (the outcome of an experiment would be the same today or tomorrow) of physical systems. The fact that physical systems behave the same, regardless of where they are located or how they are oriented, gives rise to the conservation of linear and angular momentum.

Symmetries are also crucial to the structure of the Standard Model of particle physics, our present theory of interactions at microscopic scales. Some are built into the model, while others appear accidentally from it. In the Standard Model, lepton flavour, the number of electrons and electron-neutrinos, muons and muon-neutrinos, and tau and tau-neutrinos, is one such conserved quantity.

[Standard Model](https://en.wikipedia.org/wiki/Physics_beyond_the_Standard_Model#/media/File:Standard_Model_of_Elementary_Particles_+_Gravity.svg)
Interestingly, in many proposed extensions to the Standard Model, this symmetry doesn’t exist, implying decays that do not conserve lepton flavour are possible. One decay searched for at the LHC is τ → μμμ (or τ → 3μ). Observation of this decay would be a clear indication of the violation of lepton flavour and a sign of long-sought new physics.

## How to help solving the problem

There are available real data from the [LHCb experiment](https://lhcb-public.web.cern.ch/) at the LHC, mixed with simulated datasets of the decay (the data are explained below in more detail). The metric used in this challenge includes checks that physicists do in their analysis to make sure the results are unbiased. These checks have been built into the solution design to help ensure that the results will be useful for physicists in future studies. 

## Resources
* [Flavour of Physics, Research Documentation](https://storage.googleapis.com/kaggle-competitions/kaggle/4488/media/lhcb_description_official.pdf)
* [Roel Aaij et al., Search for the lepton flavour violating decay τ → µµµ, 2015, JHEP, 1502:121, 2015](https://arxiv.org/pdf/1409.8548.pdf)
* [New approaches for boosting to uniformity](https://iopscience.iop.org/article/10.1088/1748-0221/10/03/T03002/pdf)

## Data
In this competition, you are given a list of collision events and their properties. You will then predict whether a τ → 3μ decay happened in this collision. This τ → 3μ is currently assumed by scientists not to happen, and the goal of this competition is to discover τ → 3μ happening more frequently than scientists currently can understand. The datasets are available at th competition's [Data Page](https://www.kaggle.com/c/flavours-of-physics/data).

It is challenging to design a machine learning problem for something you have never observed before. Scientists at CERN developed the following designs to achieve the goal.

### `training.csv`

This is a labelled dataset (the label ‘signal’ being ‘1’ for signal events, ‘0’ for background events) to train the classifier. Signal events have been simulated, while background events are real data.

This real data is collected by the LHCb detectors observing collisions of accelerated particles with a specific mass range in which τ → 3μ can’t happen. We call these events “background” and label them 0.

    FlightDistance - Distance between τ and PV (primary vertex, the original protons collision point).
    FlightDistanceError - Error on FlightDistance.
    mass - reconstructed τ candidate invariant mass, which is absent in the test samples.
    LifeTime - Life time of tau candidate.
    IP - Impact Parameter of tau candidate.
    IPSig - Significance of Impact Parameter.
    VertexChi2 - χ2 of τ vertex.
    dira - Cosine of the angle between the τ momentum and line between PV and tau vertex. 
    pt - transverse momentum of τ.
    DOCAone - Distance of Closest Approach between p0 and p1.
    DOCAtwo - Distance of Closest Approach between p1 and p2.
    DOCAthree - Distance of Closest Approach between p0 and p2.
    IP_p0p2 - Impact parameter of the p0 and p2 pair.
    IP_p1p2 - Impact parameter of the p1 and p2 pair.
    isolationa - Track isolation variable.
    isolationb - Track isolation variable.
    isolationc - Track isolation variable.
    isolationd - Track isolation variable.
    isolatione - Track isolation variable.
    isolationf - Track isolation variable.
    iso - Track isolation variable.
    CDF1 - Cone isolation variable.
    CDF2 - Cone isolation variable.
    CDF3 - Cone isolation variable.
    production - source of τ. This variable is absent in the test samples.
    ISO_SumBDT - Track isolation variable.
    p0_IsoBDT - Track isolation variable.
    p1_IsoBDT - Track isolation variable.
    p2_IsoBDT - Track isolation variable.
    p0_track_Chi2Dof - Quality of p0 muon track.
    p1_track_Chi2Dof - Quality of p1 muon track.
    p2_track_Chi2Dof - Quality of p2 muon track.
    p0_pt - Transverse momentum of p0 muon.
    p0_p - Momentum of p0 muon.
    p0_eta - Pseudorapidity of p0 muon.
    p0_IP - Impact parameter of p0 muon.
    p0_IPSig - Impact Parameter Significance of p0 muon.
    p1_pt - Transverse momentum of p1 muon.
    p1_p - Momentum of p1 muon.
    p1_eta - Pseudorapidity of p1 muon.
    p1_IP - Impact parameter of p1 muon.
    p1_IPSig - Impact Parameter Significance of p1 muon.
    p2_pt - Transverse momentum of p2 muon.
    p2_p - Momentum of p2 muon.
    p2_eta - Pseudorapidity of p2 muon.
    p2_IP - Impact parameter of p2 muon.
    p2_IPSig - Impact Parameter Significance of p2 muon.
    SPDhits - Number of hits in the SPD detector.
    min_ANNmuon - Muon identification. LHCb collaboration trains Artificial Neural Networks (ANN) from informations from RICH, ECAL, HCAL, Muon system to distinguish muons from other particles. This variables denotes the minimum of the three muons ANN. min ANNmuon should not be used for training. This variable is absent in the test samples.
    signal - This is the target variable for you to predict in the test samples.

### `test.csv`

The test dataset has all the columns that training.csv has, except mass, production, min_ANNmuon, and signal. 

The test dataset consists of a few parts:

    simulated signal events for the τ → 3μ
    real background data for the τ → 3μ
    simulated events for the control channel, (ignored for scoring, used by agreement test)

    real data for the control channel (ignored for scoring, used by agreement test)

You need to submit predictions for ALL the test entries. You will need to treat them all the same and predict as if they are all the same channel's collision events. 

A submission is only scored after passing both the agreement test and the correlation test. 
check_agreement.csv: Ds → φπ data

This dataset contains simulated and real events from the Control channel Ds → φπ to evaluate your simulated-real data of submission agreement locally. It contains the same columns as test.csv and weight column. For more details see agreement test.
check_correlation.csv

This dataset contains only real background events recorded at LHCb to evaluate your submission correlation with mass locally. It contains the same columns as test.csv and mass column to check correlation with. For more details see correlation test.


You can use the [`kaggle api`](https://github.com/Kaggle/kaggle-api) to donwload the dataset (you will need to have a kaggle account and to agree with the [competition](https://www.kaggle.com/c/flavours-of-physics/overview)'s agreement to use this):
```sh
kaggle competitions download -c flavours-of-physics
```




 
