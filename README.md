# SFCMCP: Semi-supervised Fuzzy Conformal Prediction for Data Stream Classification

This repository contains the source code for the method proposed in the paper:

**"Semi-supervised Fuzzy Conformal Prediction for Data Stream Classification"**

SFCMCP is a semi-supervised learning framework designed for classification in evolving data streams. The method combines semi-supervised fuzzy clustering, conformal prediction, and concept drift detection to incrementally update the classification model under limited labeled data.

The implementation is provided as a Jupyter Notebook.

---

## Repository Structure

```text
SFCMCP/
│
├── SFCMCP.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

- `SFCMCP.ipynb`: Main implementation of the proposed SFCMCP method.
- `requirements.txt`: Python packages required to run the implementation.
- `README.md`: Description and instructions for running the code.
- `LICENSE`: License information for the source code.

---

## Main Components

The implementation includes the main components of the SFCMCP framework:

- Semi-supervised fuzzy c-means clustering
- Fuzzy membership estimation
- Micro-cluster construction and incremental updating
- Micro-cluster summarization
- Conformal prediction based on fuzzy non-conformity measures
- Concept drift detection using the Kolmogorov-Smirnov test
- Incremental classifier updating
- Classification of evolving data streams under limited supervision

---

## Requirements

The implementation requires Python and the following packages:

```text
numpy
pandas
scipy
scikit-learn
scikit-multiflow
```

The required packages can be installed using:

```bash
pip install -r requirements.txt
```

It is recommended to use the same Python and package versions used for the experiments when reproducing the reported results.

---

## Dataset Format

The input dataset should be provided as a CSV file.

The current implementation assumes that:

- Each row represents one data sample.
- Feature values are stored in the preceding columns.
- The class label is stored in the last column.
- Features are represented numerically.
- Class labels are integer encoded.

For compatibility with the current implementation, class labels should preferably be encoded as consecutive integers starting from zero, for example:

```text
0, 1, 2, ..., K-1
```

where `K` is the number of classes.

Before running the notebook, specify the path to the dataset in the data-loading section:

```python
DATA_PATH = "path/to/dataset.csv"
dataframe = read_csv(DATA_PATH)
```

---

## Experimental Parameters

The main parameters of SFCMCP can be configured near the beginning of the notebook.

The default settings used in the provided implementation include:

```python
chunk_size = 5000

# Semi-supervised fuzzy clustering parameters
m = 2
MAX_ITERS = 100
sigma = 0.00001

lambda1 = 1
lambda2 = 0.1

# Proportion of labeled samples in each chunk
Percentage = 0.05 or 0.1

# Membership-related parameter
Mean_memb = 0.4

# Conformal prediction parameter
significance_level = 0.90
```

Additional parameters used during online stream processing include:

```python
max_MC = 30
Memb_th = 0.7
```

The parameters have the following roles:

- `chunk_size`: Number of samples processed in each data chunk.
- `m`: Fuzziness coefficient used in fuzzy clustering.
- `MAX_ITERS`: Maximum number of clustering iterations.
- `sigma`: Convergence threshold for the clustering objective function.
- `lambda1`: Weight of the first term of the semi-supervised clustering objective.
- `lambda2`: Weight of the supervision-related term of the clustering objective.
- `Percentage`: Proportion of labeled samples considered in each data chunk.
- `Mean_memb`: Membership threshold used during micro-cluster updating.
- `significance_level`: Threshold used in the conformal prediction procedure.
- `max_MC`: Maximum allowed number of micro-clusters per class.
- `Memb_th`: Minimum membership confidence used when updating the model with unlabeled samples.

These parameters can be modified according to the dataset and experimental setting.

---

## Running the Code

### 1. Clone the repository

```bash
git clone https://github.com/<USERNAME>/SFCMCP.git
cd SFCMCP
```

Replace `<USERNAME>` with the GitHub username hosting this repository.

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Open the Jupyter Notebook

Using Jupyter Notebook:

```bash
jupyter notebook SFCMCP.ipynb
```

Alternatively, the notebook can be opened using JupyterLab, Visual Studio Code, Google Colab, or another environment that supports `.ipynb` files.

### 4. Specify the dataset path

Modify the data-loading section of the notebook:

```python
DATA_PATH = "path/to/dataset.csv"
dataframe = read_csv(DATA_PATH)
```

### 5. Configure the experimental parameters

Set the desired values for parameters such as:

```python
chunk_size
Percentage
lambda1
lambda2
max_MC
Memb_th
```

### 6. Run the notebook

Execute the notebook cells sequentially from top to bottom.

---

## Processing Procedure

The implementation follows a chunk-based data stream processing procedure.

### Model Initialization

The first data chunk is used to initialize the learning model.

A limited proportion of the samples is selected as labeled data, while the remaining samples are treated as unlabeled. The labeled samples are used to initialize the classifier, and the semi-supervised fuzzy clustering procedure is then applied to construct the initial micro-clusters.

The clustering results are summarized using the statistics required for subsequent incremental updates.

### Online Processing

After initialization, incoming samples are processed chunk by chunk.

For each new chunk, the implementation performs the following main operations:

1. Posterior probabilities are estimated for incoming samples.
2. Fuzzy membership degrees with respect to the maintained micro-clusters are calculated.
3. Conformal non-conformity measures and p-values are computed.
4. The distributions of conformal p-values from consecutive chunks are compared.
5. The Kolmogorov-Smirnov test is used to identify possible changes in the data distribution.
6. Existing micro-clusters are incrementally updated when appropriate.
7. New micro-clusters are constructed when a concept change is detected.
8. Micro-cluster statistics and weights are maintained throughout stream processing.

The micro-cluster representation allows the model to summarize previously learned information without retaining all historical stream samples.

---

## Semi-supervised Fuzzy Clustering

The implementation includes the semi-supervised fuzzy clustering procedure used by SFCMCP.

The main functions associated with this component include:

```text
initializeClusterCenters()
ComputeF()
ComputeS()
initializeMembershipWeights()
updateMembershipWeights()
computeCentroids()
Loss_calc()
SSFCM()
```

These functions construct the supervision and membership matrices, update fuzzy memberships, calculate cluster centers, and iteratively minimize the clustering objective function.

---

## Micro-cluster Representation and Updating

The micro-cluster model stores summary statistics that enable incremental updates without requiring all previously observed samples.

The main related functions include:

```text
summarize()
update_statistics()
Train()
New_model_construction()
```

The stored statistics are updated as new labeled and unlabeled samples become available.

The number of maintained micro-clusters is controlled to avoid unlimited model growth during long-running data streams.

---

## Conformal Prediction

Conformal prediction is used to evaluate the conformity of incoming samples with respect to the learned concepts.

The main related functions are:

```text
NCM()
Prediction_by_CP()
```

The non-conformity measure is derived from fuzzy membership information. The resulting conformal p-values are subsequently used in the concept drift detection mechanism.

---

## Concept Drift Detection

Concept drift is detected by monitoring changes in the distributions of conformal p-values between consecutive data chunks.

The implementation applies the two-sample Kolmogorov-Smirnov test:

```python
sp.stats.ks_2samp(...)
```

When the resulting test indicates a statistically significant change, the model updates its representation by constructing new micro-clusters from the currently available labeled data.

Otherwise, the existing classifier and micro-clusters are incrementally updated.

---

## Output

During execution, the notebook reports classification accuracy for the processed data chunks and calculates the overall average accuracy.

The implementation can also save information related to:

- Classification accuracy
- Processing time
- Kolmogorov-Smirnov test values
- Detected distribution changes

---

## Code Availability

This repository provides the implementation of the SFCMCP framework described in the corresponding manuscript. The source code is made publicly available to facilitate reproducibility, further evaluation, and future research on semi-supervised classification of evolving data streams.

---

## Citation

If you use this implementation in your research, please cite the corresponding paper:

**N. Samadi, J. Tanha, and M. Jalili, "Semi-supervised Fuzzy Conformal Prediction for Data Stream Classification."**

The complete citation information will be updated after publication.

---

## License

The source code is distributed under the terms specified in the `LICENSE` file included in this repository.

---

## Contact

For questions regarding the implementation or the proposed SFCMCP method, please contact the first author using the below contact information:
Email: n.samadi[at]phd.tabrizu.ac.ir (N. Samadi)
