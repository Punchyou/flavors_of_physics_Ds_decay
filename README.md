# Flavors of Physics, The Strange D Meson Decay
Welcome to Flavors of Physics, The Strange D Meson Decay Solution. This project is about providing a model that solves the binary classification problem of whether a specific Ds meson decay happens in a collision event, using a machine learning algorithm.

## Problem Definition
### Domain Background

This project is a particle physics problem. Its name is inspired by what physicists call "[flavor](https://en.wikipedia.org/wiki/Flavour_(particle_physics))", the species of an elementary particle. The  [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) of particle physics is a well-established theory that explains the properties of fundamental particles and their interactions, describing the "flavor" of each particle. As mentioned in Charlotte Louise Mary Wallace CERN [Thesis](https://cds.cern.ch/record/2196092/files/CERN-THESIS-2016-064.pdf), the Standard Model theory has been tested by multiple experiments, but despite its successes, it is still incomplete, and further research is needed. 

The Standard Model counts six flavors of quarks and six flavors of leptons, as shown below. "Flavor" is essentially a [quantum number](https://en.wikipedia.org/wiki/Flavour_(particle_physics)#Quantum_numbers) that characterizes the quantum state of those quarks.

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Standard_Model_of_Elementary_Particles.svg/1024px-Standard_Model_of_Elementary_Particles.svg.png" alt="drawing" width="300"/>
</div>

 The Ds decay project is influenced by a CERN [kaggle competition problem](https://www.kaggle.com/c/flavours-of-physics/overview/description) about the flavors of physics. In the initial problem, scientists try to find if it is possible the τ (*tau*) lepton to [decay](https://en.wikipedia.org/wiki/Particle_decay) (transform into multiple other particles) to three μ (muon) leptons. The problem I chose, however, concerns the [Ds meson](https://en.wikipedia.org/wiki/D_meson) or *strange D meson*, a composite particle that consists of one quark or one antiquark, and how often it decays into a *φ* ([phi meson](https://en.wikipedia.org/wiki/Phi_meson)) and a *π* ([pi meson or pion](https://en.wikipedia.org/wiki/Pion)) based on multiple factors. The decay is described by the following flow:


<div align="center" style="text-align:center">
<img src="https://latex.codecogs.com/svg.latex?\color{blue}\Large&space;D_{s}\to\phi\pi" title=""/>
</div>


You can see where the meson belongs in the subatomic particles map below. The purple part describes the composite particles.

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Particle_overview.svg/1920px-Particle_overview.svg.png" alt="drawing" width="400"/>
</div>

Ander Ryd in his [paper](https://wiki.classe.cornell.edu/pub/People/AndersRyd/DHadRMP.pdf) argues that the D meson decays have been a challenge, though scientists have been focused on their decays since the particle discovery. As a result, the existing dataset of this project is sufficient and based on well-studied experiment observations.


### Problem Statement

The problem falls into the category of binary classification problems. Based on particle collision events (that cause the *Ds* to *φπ* decays) and their properties, I am challenged to train a machine learning model that predicts whether the decay we are interested in happens in a collision.

You can read more about the project solution in the [project report](https://github.com/Punchyou/flavors_of_physics_Ds_decay/blob/master/project_report.pdf).

## Project Structure
```git
.

├── README.md
├── data
│   ├── resampled_data.csv <-- dataset used throughout the project 
│   └── resampled_data.csv.zip <-- resample dataset as .zip
├── data_exploration.py <-- script for the project data analysis
├── images
│   ├── accuracy_heatmap.png <-- accuracy score heatmap for all the models
│   ├── comparison_heatmap.png <-- heatmap for all the metrics values between benchmark and final model
│   ├── features_correlation_heatmap.png <-- Pearson correlation coefficients between all the features in the dataset
│   ├── features_distributions.png <-- histograms for all the features in the dataset
│   ├── knn_benchmark_acc.png <-- accuracy score scatterplot for range of k parameters of kNN
│   ├── learning_curve.png <-- learning curve plot for the final XGBoost model
│   ├── pca_binary_scatter_3d_plot.png <-- 3D plot of the 3 principal components of the scaled dataset
│   ├── project_ds_workflow.png <-- data science workflow (used in project proposal)
│   └── signal_value_counts.png <-- simple count barplot with the number samples of each class
├── knn_benchmark_model.py <-- script with the kNN benchmark model
├── model.py <-- script for the final optimized XGBoost model and learning curve analysis
├── models_exploration.py <-- script with all the models trained for the project
├── project_report.md <-- background, analysis and results for this project
├── project_report.pdf <-- project report in pdf format
├── proposal.md <-- project proposal
├── proposal.pdf <-- project proposal in pdf format
├── reports
│   ├── XGBClassifier_cv_results.csv <-- all XGBoost models as calculated from the Bayesian optimization algorithm
│   ├── metrics_results.csv <-- All metrics for all the models
│   ├── model_final_metrics.csv <-- Metrics for the final XGBoost model
│   └── signal_final_prediction.csv <-- predictions for the test set
├── requirements.txt <-- requirements to be installed to execute the modules of the project
└── utils.py <-- utility functions for the project (statistics, metrics and visualization functions)
```
## Get the project
Clone the project through HTTPS:
```git
git clone https://github.com/Punchyou/flavors_of_physics_Ds_decay.git
```

Or if you have added an SSH key to GitHub:
```git
git clone git@github.com:Punchyou/flavors_of_physics_Ds_decay.git
```

## Execute the project files
Note that all the plots and report files that are generated by the python files, already exist under `images/` and `reports/` respectively. Refer to the [project report](https://github.com/Punchyou/flavors_of_physics_Ds_decay/blob/master/project_report.pdf) (Data Exploration, Algorithms implementation, Models Performance and Conclusion sections) for more.

If you want to execute the files that generate the plots and the report `.csv` files, follow the instructions below. After running the python files, the existing images and reports will be replaced with the newly generated ones.

### Pre-requests:
* It is recommended to use a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or a [python `virtualenv` environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/), to install all the required packages there.
* Install requirements by running `pip install -r requirements.txt`. *Make sure that you have `sklearn 0.23.2` installed by running `pip freeze`.*

### Execute the scripts:
1. To generate the plots from the data exploration analysis (`features_correlation_heatmap.png`, `features_distributions.png`, `knn_benchmark_acc.png` under `images/`) you can execute:
```sh
python data_exploration.py
```
> *The line that generates `pca_binary_scatter_3d_plot.png` is currently commented out in `data_exploration.py`, as it is a 3D plot and the currently saved image is the optimal angle for the best view of the two classes. The commented line saves that plot in a different angle, which is not ideal to have an understanding of the data in the 3D space.*

2. Execute the benchmark model script and generate the `knn_benchmark_acc.png` plot, with the accuracy of the model for different `k` parameters values:
```sh
python knn_benchmark_model.py
```

3. For generating the performance metrics file for all the models trained for this project (`reports/metrics_results.csv`), and the accuracy heatmap for all the models (`images/accuracy_heatmap.png`), run:
```sh
python models_exporation.py 
```
> *This runs all the models and the hyperparameter tuning algorithms, so it will take several minutes to complete.*


4. Execute the final model script and generate the `images/learning_curve.png` and `images/comparison_heatmap.png` for the comparison with the benchmark model, and also the `reports/model_final_metrics.csv` and `reports/signal_final_prediction.csv`.
```sh
python model.py
```