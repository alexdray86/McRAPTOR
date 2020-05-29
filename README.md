# McRAPTOR based delay-predicting robust journey planner

A. Coudray, F. Grimberg, C. Pulver, T. Turner


**Please [watch our video presentation](https://youtu.be/nVHRUHIp0BA)**


**Please view this file in GitLab so the formulas are displayed correctly.**

This repository represents the final project of group **lgptguys** for the spring 2020 _Lab in Data Science_ course at EPFL. Find the original assignment for the project in [Project_Description.md](Project_Description.md).

Our work consists of a robust journey planner, which finds journeys between stops within a 15km radius around Zurich Hbf and computes their probability of success based on the distribution of delays over the past years.
Given a source stop, a target stop, a latest acceptable arrival time, and a lowest acceptable probability of success for the entire journey, our journey planner finds the journey whose departure time is latest among all acceptable options.
Our journey planner is based on the multiple-criteria (Mc) extension of the RAPTOR algorithm (Round bAsed Public Transit Optimized Router) detailed in [Delling, D., Pajor, T., & Werneck, R. F. (2015). Round-based public transit routing. *Transportation Science, 49*(3), 591-604.](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf).


### Content
1. [Using the files in this repo](#1.-Using-the-files-in-this-repo)
2. [Choosing a route planning algorithm](#2.-Choosing-a-route-planning-algorithm)
3. [Modeling the public transit infrastructure](#3.-Modeling-the-public-transit-infrastructure)
4. [Building a predictive model for the success of a journey](#4.-Building-a-predictive-model-for-the-success-of-a-journey)
5. [Implementing the route planning algorithm](#5.-Implementing-the-route-planning-algorithm)
6. [Conclusion, Outlook](#6.-Conclusion,-Outlook)

### Team and attributions:

- Journey planning algorithm selection, implementation, adaptation and improvement: **Felix Grimberg, Cyril Pulver**
- Modeling the probabilities of delays and missed connections (includes the data wrangling for this part): **Alexandre Coudray**
- Data wrangling from the SBB timetable to RAPTOR data structures: **Cyril Pulver, Tomas Turner**
- Editing for the video presentation: **Tomas Turner**

## 1. Using the files in this repo

### Finding a journey from A to B

To use our robust journey planner, simply open [our McRAPTOR notebook](notebooks/MC_RAPTOR.ipynb) and follow the instructions outlined at the start of the notebook.

### Reproducing our work

To reproduce all of our work, starting from the available SBB data, or to dive into our work more in detail, please follow these steps:

1. Transform the GTFS files for the Swiss transport network to GTFS-like files `stop_times` and `transfers` by running the [data wrangling before RAPTOR notebook](notebooks/data_wrangling_before_RAPTOR.ipynb) in its entirety.
2. Pull the `.csv` HDFS files from the distributed file system to the local Renku environment by executing the [transfer to local notebook](notebooks/transfer_to_local.ipynb)
3. Transform the `.csv` GTFS-like tables to python objects usable within our implementation of RAPTOR using the [generating arrays for RAPTOR notebook](notebooks/generating_arrays_for_RAPTOR.ipynb)
4. Generate `match_datasets_translation.csv` HDFS file from [hdfs_match_datset notebook](notebooks/hdfs_match_datasets.ipynb)
5. Generate dictionnaries of delay distribution from hdfs and save it to local in pickles : `data/d_all.pck.gz` and `data/d_real.pck.gz` from [hdfs_get_distribution notebook](notebooks/hdfs_get_distributions.ipynb)
6. Generate pre-computed probability of success 2d numpy (McRAPTOR input) `transfer_probabilities.pkl.pkl.gz` from distributions in [gen_transfer_proba notebook](notebook/gen_transfer_proba.ipynb)
7. Then, open [the McRAPTOR notebook](notebooks/MC_RAPTOR.ipynb) and follow steps 2-4 at the start of the notebook.

## 2. Choosing a route planning algorithm

We investigated several options, such as implementing a custom Breadth-First Search (BFS) or Dijkstra algorithm on a time-expanded graph.
However, after a brief literature search, we decided to adapt and implement the multiple criteria variant (McRAPTOR) of the state-of-the-art algorithm described in [Delling et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf) (cf. complete citation above).
The main arguments in favor of this algorithm are as follows:
- RAPTOR (and its variants) does not require transforming the public transit schedule into a (time-extended or time-expanded) graph.
Instead, the algorithm directly takes advantage of the nature of public transit networks, where trips occur only along certain routes and at certain times of day.
This is much more efficient, as it avoids the crass increase in graph size linked to time-expansion. Our implementation of McRAPTOR runs on a single core and a very modest amount of RAM.
The RAPTOR approach also more closely mimicks actual passenger behaviour, which tends to fully explore the currently available routes before exploring possible transfers between routes.
- The paper proposes a generic multiple-criteria variant based on notions of [Pareto efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency), primarily Pareto domination and Pareto sets.
This variant can be adapted to include any number of arbitrary optimization criteria at the cost of a slightly less efficient implementation.
However, most of the increase in execution times from RAPTOR to McRAPTOR can probably be attributed to the inescapable fact that multiple-criteria optimization is a harder problem than single-criteria optimization, simply because it admits more solutions at each step along the way.

## 3. Modeling the public transit infrastructure
RAPTOR models the public transit infrastructure similarly to a standard GTFS timetable. The "atoms" of this representation are the stop times (stored in the GTFS-like array StopTimes): an arrival and a departure time record for each and every time a vehicle stops to pick up / drop off passengers somewhere in the transport network. RAPTOR uses a collection of arrays storing pointers corresponding to routes, trips and stops to explore the StopTimes array efficiently. In particular, RAPTOR makes use of in-memory data locality when exploring the arrays. This guarantees fast and light-weight journey planning.

### The SBB timetable dataset
The raw material to model the transport network is the GTFS-formatted SBB timetable corresponding to the week of May 14th 2019. We consider the Zürich transport network only, on standard business days, between 7am and 8pm, for services that run each day of the week.

### Producing the data structures required to run RAPTOR
We formatted the GTFS file `stop_times.txt` to a cleaned GTFS-like `stop_times` table which directly corresponds to the StopTimes array at the core of RAPTOR. To ensure a coherent file structure within RAPTOR, trips and routes were reconstructed directly from the cleaned `stop_times` table. In depth-explanations of the data wrangling process to model the transport network in RAPTOR can be found in the [data wrangling before RAPTOR notebook](notebooks/data_wrangling_before_RAPTOR.ipynb) and [generating arrays for RAPTOR notebook](notebooks/generating_arrays_for_RAPTOR.ipynb).

### Validation / testing / evaluation
We checked that trips found in the GTFS-like `stop_times` table corresponded to real-life trips proposed in the SBB timetable (sbb.ch).

RAPTOR makes use of a pointer/array structure. Thus a single undesired shift in one of the arrays would compromise the whole algorithm. Therefore, we performed regular sanity checks to assess consistency between arrays and pointers. When relevant, we employed several ways of computing the latter from the former and tested the equality of the results. One example may be found in the [generating arrays for RAPTOR notebook](notebooks/generating_arrays_for_RAPTOR.ipynb) at cells 30-35.

In a final test phase, we used the produced datasets to find journeys between random stops with our robust journey planner and checked each walking path and each transport separately in Google Maps and the SBB website respectively. This approach proved very useful to correct for several bugs.

## 4. Building a predictive model for the success of a journey

To estimate the probability of success of a given transfer, our strategy is to maximize the usage of historical data, to get quasi assumption-free probability estimates. 

One of the input of our McRaptor implementation is a 2-dimensionnal array of pre-computed probabilities, leveraged from past delays in the *SBB istdaten* dataset. We pre-computed probability of success for every combinations of `trip_id` and `stop_id` (~250k combinations for stops in 15km perimeter around Zurich HB). These are then used in McRaptor to choose the best trip not only based on the latest departure time, but also on their **probability of success**.

Our aim is to optimize the use of historical data given any possible input parameters, allowing us to capture probabilities in all cases.

### Matching SBB istdaten and Timetable datasets 

_Note : This section summarize what is done in [hdfs_match_datasets.ipynb](notebooks/hdfs_match_datasets.ipynb)_

To be able to compute probability of success for a given transfer, we rely entirely on past data using delays from *SBB istdaten*. To be able to do that, we need a clear and robust translation between _timetable_ and _sbb_ data, as the trip_id do not match between both datasets. This a short summary of what was done to link both datasets :

__stop_id :__ _timetable_ `stop_id` may contains additional information about platform, and therefore need to be trimmed to its first 7 characters to match _sbb_ `bpuic` id.

__trip_id :__ To match both datasets, we matched `stop_id` , `departure_time` and `arrival_time` (with a join) to get corresponding trip_id between datasets. The idea is to take every trip_id with more than X matches between both datasets. 

_Note : We decided to use 2 as a minimum number of matches needed, as it was the only way to get Intercity / Eurocity trains that often have only two stops in the 15km perimeter._

### Get Distributions of Delay Times per trip and station

_Note : This summarize [hdfs_get_distributions.ipynb](notebooks/hdfs_match_datasets.ipynb)_

Using data generated in [hdfs_match_datasets.ipynb](notebooks/hdfs_match_datasets.ipynb), we can compute arrival delays from _sbb_ dataset and match it with _timetable_ `trip_id`. For each given `trip_id` x `stop_id`, we generate an array of delays from -1 (contains all trips ahead of schedule) to +30 (also contains trips being more than 30 minutes late).

We generate two tables of arrival delay distribution per `trip_id`x`stop_id` : once for `geschaetz` / `real` delays only, and once for `all` delays. Delays in `geschaetz/real` group are used in priority if there is enough data. 

### Compute probability of transfer success from delays distributions

_Note : Can be found in [gen_transfer_proba.ipynb](notebooks/gen_transfer_proba.ipynb)_

To be able to compute the probability of success of a given transfert, we use the arrival delay distribution compared with the next trip departure. We entirely rely on past data and therefore on assumption-free and model-free probabilites.

To be able to do that, we need delay distributions for each trip arrival to a given station. We then use a __cumulative distribution function__ to compute $`P(X \leq x)`$ :

$${\displaystyle F_{X}(x)=\operatorname {P} (T\leq t)=\sum _{t_{i}\leq t}\operatorname {P} (T=t_{i})=\sum _{t_{i}\leq t}p(t_{i}).}$$

The strategy was to rely entirely on past data to compute $`p(t_i)`$, without the need of building a model based on additionnal assumptions. If we have enough data for a given transfer with known `trip_id` x `stop_id`, we use the the abovementionned formula to compute each $`p(t_i)`$ by simply using :

$$p(t_i) = \frac{d_i}{\sum x_i}$$

with $`d_i`$ being the number of delays at time $`t_i`$ from SBB dataset and $`\sum x_i`$ the total number of trips arriving at the stop at time $`t_i`$.

We make a few __assumptions__ :
- We assume that if we have less than 2 minutes for the transfer, we miss it.
- We assume the next train is on time.

### Recover missing data 

Sometimes we may have few data points or a parameter missing. To recover these cases, we generate aggregated delay distributions over 4 parameters that characterize a transfer. We used an approach we called “cascades of recovery distributions”, which define how we leverage the parameters at disposal to compute a probability. The idea is to use increasingly aggregated data to optimize the use of historical data given any possible input parameter combination.

We use 4 different parameters, `stop_id`, `transport_type`, `trip_id` and `time` to aggregate delays and compute probabilities of transfer success leveraging data from similar transfers. For instance, whenever `trip_id` information is missing, we may rely aggregated delay based on 3 other parameters, which gives a good approximation of delay in function of space / time and type of transporation.

### Evaluate / Validate recovery tables

We validated this approach and defined a hierachy of recovery distribution usage by comparison with actual measures from sbb dataset.

The strategy was to first evaluate how precise these recovery table are by using `geschaetz/real` data with > 100 points and compare the aggregated delay distribution with the actual delays given a `trip_id`x`stop_id`, which allowed us to compute residual error in probability inference.

Based on these results, we established a hierarchy of recovery tables, each of them being used in a _cascade_, meaning whenever one of them succeed, next recovery table are not used. Using this method, every possible transfer with missing parameter can be given a probability of success.

## 5. Implementing the route planning algorithm

[Delling et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf) (cf. complete citation above) give a pseudocode of the RAPTOR algorithm, which solves the earliest arrival problem while simultaneously optimizing for the lowest possible number of individual trips in a journey.

### Solving the latest departure problem

We first implemented the algorithm in Python, following the pseudocode exactly, and tested it on a miniature toy dataset.
Then, we inverted it to solve the **latest departure problem**: We begin by marking the target stop, we traverse routes "backwards", we flip comparisons, and we initialize the value for each stop with 0 instead of $`\infty`$.

### Optimizing for the probability of success of a journey

Delling et al. also describe how to change the RAPTOR algorithm into a variant with multiple criteria.
Instead of a single arrival/departure time per round, each stop can now have an arbitrary number of [Pareto optimal](https://en.wikipedia.org/wiki/Pareto_efficiency) solutions ("label") per round.
These labels are stored in bags, and the most relevant operation is to merge a label (or another bag) into an existing bag, discarding all Pareto dominated labels in the process.
We chose to implement these labels with a hierarchy of object-oriented label classes as explained in [the McRAPTOR notebook](notebooks/MC_RAPTOR.ipynb), to store the optimization criteria (departure time, probability of success, and number of trips), provide functionality (e.g., testing for Pareto domination), and more.
While each label represents a journey from a given stop to the destination stop, they only contain information relevant to one trip or walk within that solution (except for the optimization criteria, which pertain to the entire "rest" of the journey from the given stop).
Importantly, each label also contains a reference to the next label, so that the entire journey can be reconstructed recursively.
The probability of success associated with each label is also constructed recursively, by multiplying the probability of success of the next label with the probability of succeeding in the connection between this label and the next.

### Additional adjustments

Since the user is asked to specify a minimum acceptable overall probability of success for their journey, any label can be discarded if its probability of success does not meet the bar.
This form of *constraint pruning* can potentially save a lot of computational effort downstream.
It is implemented exactly in the same way as target pruning (cf. [Delling et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/raptor_alenex.pdf)).
Finally, we had to account for some tricky details, such as how to traverse a route if it visits the same stop multiple times.

### Validation / testing

We wrote helper functions to test the journey planner on random start and destination stops, carefully checking the printed solutions and comparing them to the journeys suggested on [the SBB website](https://www.sbb.ch/en/buying/pages/fahrplan/fahrplan.xhtml).
Our journey planner more readily suggests walking between stops (though never twice in a row, and never more than 500m / 10min) than the SBB journey planner, especially if there are obstacles or rivers between the two stations.

## 6. Conclusion, Outlook

Our robust journey planner is only a proof of concept that illustrates what robust journey planning would be like given additional ressources and development efforts. In particular, the following points require further work:

- Extending the journey planner beyond Zürich
- Adapting the algorithm for parallel computing
- Input of user-friendly stop names instead of abstract stop id
- Starting with GPS coordinates as input/output
- Accounting for the departure delays when computing the probability of success of a single connection
- Considering the dependence of delays
- Computing probabilities of delays based on real-time data
- Using a more realistic approach for foot-paths
- Using a more realistic model for changing platforms

Our robust route planning algorithm **efficiently** finds the set of **all Pareto optimal journeys**, to get from one stop to another within constraints of **latest acceptable arrival time** and **minimum acceptable probability of success**.
It is based on an assumptions-free, tiered delay probability model that always makes the most of the available data (i.e., probabilities are more accurate wherever more accurate data is present).
