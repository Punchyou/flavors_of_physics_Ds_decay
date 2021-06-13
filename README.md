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

$$D_s \to φπ$$

You can see where the meson belongs in the subatomic particles map below. The purple part describes the composite particles.

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Particle_overview.svg/1920px-Particle_overview.svg.png" alt="drawing" width="400"/>
</div>

Ander Ryd in his [paper](https://wiki.classe.cornell.edu/pub/People/AndersRyd/DHadRMP.pdf) argues that the D meson decays have been a challenge, though scientists have been focused on their decays since the particle discovery. As a result, the existing dataset of this project is sufficient and based on well-studied experiment observations.


### Problem Statement

The problem falls into the category of binary classification problems. Based on particle collision events (that cause the $D_s \to φπ$ decays) and their properties, I am challenged to train a machine learning model that predicts whether the decay we are interested in happens in a collision. The model will be trained on the training set, which the 80% of the existing dataset described in the section, and it will be evaluated on the remaining data.


## Project Structure
```
.

├── README.md
├── data
│   ├── resampled_data.csv <-- dataset used throught the project 
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
│   ├── XGBClassifier_cv_results.csv <-- all XGBoost models 
│   ├── metrics_results.csv <--
│   ├── model_final_metrics.csv <--
│   └── signal_final_prediction.csv <--
├── requirements.txt <--
└── utils.py <--
```
## TODO: Clone the project
TODO: mention that I have commented out the 3d plot
## How to execute the files
Note that all the plots and `.csv` report files are generated by the python files already exist under `images/` and `reports/` respectively. Please refer to the `project_report.md` (sections Data Exploration and *Models' Performance*)for more information.

If you want to execute the files that generate the plots and the report `.csv` files, follow the instructions below.
1. Pre-requests:
* It is recommended to use a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or a [python `virtualenv` environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/), to install all the required packages there.
* Install requirements by running `pip install -r requirements.txt`. Make sure that you have `sklearn 0.23.2` installed

2. Execute the scripts:
* If you want to generate the plots from the data exploration analysis, you can execute:
```py
python data_exploration.py
```
* Generate the plots TODO: Learning Curve
This will generate new and the plots, and the existing ones will be replaced.
* For generating the performance metrics
