<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
# Flavors of Physics: Meson Decay Project Proposal

## Domain Background


This project is a particle physics problem. Its name is inspired by what physicists call "[flavor](https://en.wikipedia.org/wiki/Flavour_(particle_physics))", the species of an elementary particle. The  [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) of  particle  physics  is  a  well  established  theory that explains the properties of fundamental particles and their interactions, describing the "flavor" of each particle. As mentioned in Charlotte Louise Mary Wallace CERN [Thesis](https://cds.cern.ch/record/2196092/files/CERN-THESIS-2016-064.pdf), the Standard Model theory has been tested by multiple experiments, but despite its successes, it is still incomplete and further research is needed. 

The Standard Model counts six flavors of quarks and six flavors of leptons, as shown below.

![standard_model](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Standard_Model_of_Elementary_Particles.svg/1024px-Standard_Model_of_Elementary_Particles.svg.png)


"Flavor" is essentially a [quantum number](https://en.wikipedia.org/wiki/Flavour_(particle_physics)#Quantum_numbers) that characterizes the quantum state of those quarks. This project is influenced by a CERN [kaggle competition](https://www.kaggle.com/c/flavours-of-physics/overview/description) problem about those flavors of physics. In the initial problem, scientists try to find if it is possible that the τ (tau) lepton to [decay](https://en.wikipedia.org/wiki/Particle_decay) (transform into multiple other particles) to three μ (muon) lepton (both shown in the standard model image above). The current problem however concerns the [Ds meson](https://en.wikipedia.org/wiki/D_meson) (strange D meson), a composite particle that consists of one quark or one antiquark, and how often it decays into a φ ([phi meson](https://en.wikipedia.org/wiki/Phi_meson)) and a π ([pi meson or pion](https://en.wikipedia.org/wiki/Pion)) based on multiple features and observations. The decay is described below:

$DS -> φπ$

You can see where the meson belongs in the subatomic particles map below. The purple part describes the composite particles.

![composite_particles](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Particle_overview.svg/1920px-Particle_overview.svg.png)

Ander's Ryd in his [paper](https://wiki.classe.cornell.edu/pub/People/AndersRyd/DHadRMP.pdf) argues that the Ds decays have been a challenge, though scientists have been focused on their decays since the particles discovery. As a result the existing dataset of  this project ia sufficient and based on well-studied experiment observations.


## Problem Statement

The problem falls into the category of binary classification problems. Based on particle collision events that cause the Ds decay and their properties, I am challenged to predict whether a decay happens in a collision or not.

## Datasets and Inputs

As described in the [flavors of physics](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test) project, the TODO Ds -> fipi decay has a very similar topology as the TODO tau -> mmm decay and their datasets share almost the same features. In the tau decay problem, the Ds decay data is used as part of the the CERN evaluation process for that problem. This dataset will be used as the main dataset of the TODO Ds->fipi decay problem solution.

This is a labelled dataset (the label ‘signal’ being ‘1’ for decays happening (signal events) and ‘0’ for decays not happening (background events)) to train the classifier.

* FlightDistance - Distance between Ds and PV (primary vertex, the original protons collision point).
* FlightDistanceError - Error on FlightDistance.
* LifeTime - Life time of Ds candidate.
* IP - Impact Parameter of Ds candidate.
* IPSig - Significance of Impact Parameter.
* VertexChi2 - χ2 of Ds vertex.
* dira - Cosine of the angle between the Ds momentum and line between PV and tau vertex. 
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
* min_ANNmuon - Muon identification. LHCb collaboration trains Artificial Neural Networks (ANN) from information from RICH, ECAL, HCAL, Muon system to distinguish muons from other particles. This variables denotes the minimum of the three muons ANN. min ANNmuon should not be used for training. This variable is absent in the test samples.
* signal - This is the target variable for you to predict in the test samples.


### Obtain the dataset
There are three ways to get the data described above:
* I recommend to download the resampled dataset from the github repo I created for this project. I intent to use this dataset, as the original is quite imbalanced. The resampled dataset is also smaller and much more easy to manage in the analysis. I made sure that the dataset have sufficient data for my analysis. If you want to get the original datasets, follow one of the next following ways.
* From kaggle, by downloading the check_agreement.csv.zip from [here](https://www.kaggle.com/c/flavours-of-physics/data?select=check_agreement.csv.zip) (this requires a kaggle account)
* Download it from my github repo I have created for this project [here](https://github.com/Punchyou/flavors_of_physics_Ds_decay/blob/master/data/resampled_data.csv.zip).


> Note that in the resampled dataset, I have dropped the "weights" feature from the original dataset, as according to the [description of the dataset](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test), is the feature used to determine if the decay happens or not based on its value, and is the one used to create the binary signal column. It will be not used in the solution whatsoever.


## Solution Statement

This is a binary classification problem so the solution will contain a binary classifier. There is no constrain in using any classifier in particular for this family of problems. However, in the [evaluation description](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test) of the kaggle competition above, is mentioned that the [Kolmogorov–Smirnov (KS)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) test is used to evaluate the differences between the classifier distribution and the true values on each sample.

 I intent to train multiple models with data from a few different cleaning methods, but in the final solution I will only present the chosen data cleaning method and model based on evaluation from different performance metrics, including the KS test.

## Benchmark Model
	
As a benchmark model I use a simple k-Nearest Neighbor classifier trained in the resampled data, and grid search for tuning the k hyperparameter. The benchmark model script can be found [here], but the code is presented below:

```py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/resampled_data.csv", index_col="Unnamed: 0")
X = df.drop("signal", axis=1)
y = df["signal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# grid search to find optimal value based on accuracy
acc = []
from_ = 1
to = 80
for i in range(from_, to):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train.values)
    pred_i = knn.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, pred_i))

# plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 80),
    acc,
    color="blue",
    linestyle="dashed",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.title("Accuracy vs. K Value")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.text(
    from_ + 1,
    max(acc),
    f"Max accuracy: {round(max(acc), 3)},  K = {acc.index(max(acc))}",
)
plt.show()
```
The execution of the code above produces the following plot. The plot show the accuracy of the kNN model for each one of the k values (from 1 to 80):

![accuracy_plot](https://raw.githubusercontent.com/Punchyou/flavors_of_physics_Ds_decay/master/images/knn_benchmark_acc.png)

## Evaluation Metrics

The evaluation metric used to choose the benchmark model has been briefly examined in the Benchmark Model Section, and it is the `sklearn` accuracy. It is calculated by dividing the number of correct predictions by the total number of samples.

## Project Design
A high level workflow for the solution approach:
1. Understand the problem - identify the problem category: I already know that this is a binary classification problem, which will help me choose the models to train.
2. Data mining: Obtain the necessary datasets for this analysis.
3. Data Visualization, Cleaning and Engineering: Check if the data are balanced, if there are linear correlation between the features to determine which features to keep or check if the data will work better with a specific scaling. Dimensionality reduction techniques may help visualize the data to have a better understanding of their shape.
4. Model Training/Tuning: Train a selection of binary classification algorithms, trying different data scaling, different number of features and cross validation to avoid overfitting. Make use of search techniques like grid or random search to tune the hyperparameters. Models like Support Vector Machines, Stochastic Gradient Descent or XGBoost classifiers might be suitable for this problem as they can deal with multiple features efficiently.
5. Model Evaluation: A number of performance metrics will be used for this step. Evaluation metrics suitable for this problem might be false positive rate, false negative rate, specificity, negative predictive value, accuracy or Kolmogorov–Smirnov test.

A workflow that sums up the steps above follows. This workflow is part of a typical data science lifecycle, as presented [here](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/lifecycle).

![project_workflow](https://raw.githubusercontent.com/Punchyou/flavors_of_physics_Ds_decay/master/images/project_ds_workflow.png)

### Why I chose this project?
I love physics and I was always curious about how and why subatomic particles behave the way they do. I have a bachelor's degree in physics, but I started having an interest in data science after my studies. I am now confident enough to start utilizing my data science knowledge in solving physics problems, combining my interest for both fields. I specifically chose a particle physics project as I know that scientists at CERN often provide real data to the public based on their observations, for that I would be able to find both an interesting problem and an associated online dataset, with sufficient data for a data science analysis.

TODO
* Mention why I chose the other meson decay problem over the lepton decay.


### Sources:
* https://en.wikipedia.org/wiki/Flavour_(particle_physics)
* [Meson Decays](https://cds.cern.ch/record/2713513?ln=en)

### Further sources:
* [Search for τ → 3μ decays using τ leptons produced in D and B meson decays](http://cds.cern.ch/record/2668282/files/BPH-17-004-pas.pdf)