# Flavors of Physics: Meson Decay Project Proposal

## Domain Background


This project is a particle physics problem. Its name is inspired by what physicists call "[flavor](https://en.wikipedia.org/wiki/Flavour_(particle_physics))", the species of an elementary particle. The  [Standard Model](https://en.wikipedia.org/wiki/Standard_Model) of  particle  physics  is  a  well  established  theory that explains the properties of fundamental particles and their interactions, describing the "flavor" of each particle. As mentioned in Charlotte Louise Mary Wallace CERN [Thesis](https://cds.cern.ch/record/2196092/files/CERN-THESIS-2016-064.pdf), the Standard Model theory has been tested by multiple experiments, but despite its successes, it is still incomplete and further research is needed. 

The Standard Model counts six flavors of quarks and six flavors of leptons, as shown below.

![standard_model](https://en.wikipedia.org/wiki/Elementary_particle#/media/File:Standard_Model_of_Elementary_Particles.svg)


"Flavor" is essentially a [quantum number](https://en.wikipedia.org/wiki/Flavour_(particle_physics)#Quantum_numbers) that characterizes the quantum state of those quarks. This project is influenced by a CERN [kaggle competition](https://www.kaggle.com/c/flavours-of-physics/overview/description) problem about those flavors of physics. In the initial problem, scientists try to find if it is possible that the τ (tau) lepton to [decay](https://en.wikipedia.org/wiki/Particle_decay) (transform into multiple other particles) to three μ (muon) lepton (both shown in the standard model image above). The current problem however concerns the [Ds meson](https://en.wikipedia.org/wiki/D_meson) (strange D meson), a composite particle that consists of one quark or one antiquark, and how often it decays into a φ ([phi meson](https://en.wikipedia.org/wiki/Phi_meson)) and a π ([pi meson or pion](https://en.wikipedia.org/wiki/Pion)) based on multiple features and observations. The decay is described below:

TODO: DS -> fipi

You can see where the meson belongs in the subatomic particles map below. The purple part describes the composite particles.

![composite_particles](https://en.wikipedia.org/wiki/Bound_state#/media/File:Particle_overview.svg)

Ander's Ryd in his [paper](https://wiki.classe.cornell.edu/pub/People/AndersRyd/DHadRMP.pdf) argues that the Ds decays have been a challenge, though scientists have been focused on their decays since the particles discovery. As a result the existing dataset of  this project ia sufficient and based on well-studied experiment observations.


## Problem Statement	

The problem falls into the category of binary classification problems. Here I am challenged to find if a TODO: Ds->fipi decay will happen or not, based on a set of observations described in the next section.

## Dataset and Inputs

As described in the [flavors of physics](https://www.kaggle.com/c/flavours-of-physics/overview/agreement-test) project, the TODO Ds -> fipi decay has a very similar topology as the TODO tau -> mmm decay and their datasets share the same features. In the tau decay problem, the Ds decay data is used as part of the the CERN evaluation process for that problem. This dataset will be used as the main dataset of the TODO Ds->fipi decay problem solution.

The datasets can be obtained with two different ways. Either from kaggle, by downloading the check_agreement.csv.zip from [here](https://www.kaggle.com/c/flavours-of-physics/data?select=check_agreement.csv.zip) (this requires a kaggle account) or download it from my github repo I have created for this project [here](TODO: add link to the project).

Remove weights: In particle physics experiments, it is well-known how to split the signal/background by their “weights”. The method is called sPlot. This weight is a measurement derived from the likelihood/probability that the event is a Ds → φπ and the weights range from -∞ to ∞. Higher weight means likely this event is signal, lower weights means it’s likely to be background. 


Solution Statement
	

Student clearly describes a solution to the problem. The solution is applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, the solution is quantifiable, measurable, and replicable.

Benchmark Model
	
A benchmark model is provided that relates to the domain, problem statement, and intended solution. Ideally, the student's benchmark model provides context for existing methods or known information in the domain and problem given, which can then be objectively compared to the student's solution. The benchmark model is clearly defined and measurable.

Evaluation Metrics

Student proposes at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model presented. The evaluation metric(s) proposed are appropriate given the context of the data, the problem statement, and the intended solution.

Project Design
	

Student summarizes a theoretical workflow for approaching a solution given the problem. Discussion is made as to what strategies may be employed, what analysis of the data might be required, or which algorithms will be considered. The workflow and discussion provided align with the qualities of the project. Small visualizations, pseudocode, or diagrams are encouraged but not required.

Presentation

Proposal follows a well-organized structure and would be readily understood by its intended audience. Each section is written in a clear, concise and specific manner. Few grammatical and spelling mistakes are present. All resources used and referenced are properly cited.

### Why I chose this project?
I love physics and I was always curious about how and why subatomic particles behave the way they do. I have a bachelor's degree in physics, but I started having an interest in data science after my studies. I am now confident enough to start utilizing my data science knowledge in solving physics problems, combining my interest for both fields. I specifically chose a particle physics project as I know that scientists at CERN often provide real data to the public based on their observations, for that I would be able to find both an interesting problem and an associated online dataset, with sufficient data for a data science analysis.

TODO
* Mention why I chose the other meson decay problem over the lepton decay.


### Sources:
* https://en.wikipedia.org/wiki/Flavour_(particle_physics)
* [Meson Decays](https://cds.cern.ch/record/2713513?ln=en)

### Further sources:
* [Search for τ → 3μ decays using τ leptons produced in D and B meson decays](http://cds.cern.ch/record/2668282/files/BPH-17-004-pas.pdf)