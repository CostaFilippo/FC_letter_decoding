# FC Letter Decoding

The main objectives of this project are:

- Perform letter decoding using population activity
- Evaluate decoding performance under different conditions
- Fit a mixed-effect glm for the response times

---

## Repository Structure

```
FC_letter_decoding/
│
├── Code/               
│   ├── 1a_localization.ipynb
│   ├── 1b_units_and_behavior.ipynb
│   ├── 2a_letter_decoding.ipynb
│   ├── 2b_singleNeuron_response_box.ipynb
│   ├── 2c_RT_INOUT.ipynb
│   ├── 3a_heldout_letter_decoding.ipynb
│   ├── 3b_response_time_change.ipynb
│   ├── 3c_area_letter_decoding.ipynb
│   ├── 3d_size_letter_decoding.ipynb
│   ├── 3e_inout_letter_decoding.ipynb
│   ├── Table1.ipynb
│   ├── compute_session_alignment.py
│   └── glm.R
│
├── Data/                # Raw and processed datasets
├── Figures/             # Generated plots and figures
└── Results.zip          # Archived analysis outputs
```

---

## Dependencies

### Python

The analysis notebooks and scripts have been tested on:

- Python: 3.8.10 
- numpy: 1.24.4
- scipy: 1.10.1
- pandas: 2.0.3
- scikit-learn: 1.3.2
- matplotlib: 3.6.0
- jupyter: 1.0.0
- statsmodels: 0.14.1
- nibabel: 5.2.1
- nilearn: 0.10.4
- pingouin: 0.5.5
- joblib: 1.3.2


### R

For statistical modeling (glm.R):

- R: 4.1.1
- lme4: 1.1.32
- lmerTest: 3.1.3
- readr: 2.1.4

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CostaFilippo/FC_letter_decoding.git
cd FC_letter_decoding
```

### 2. Install dependencies

Estimated install time: approx. 1 min

## Data&#x20;

The `Data/` directory contains the pre-processed experimental dataset: 

- Spiking data during the working memory task
- Behavioral variables (reaction time, trial type, correctness, ...)
- Session and unit metadata

---

## Analysis Pipeline

#### `1a_localization.ipynb`

- Determines anatomical locations of units

#### `1b_units_and_behavior.ipynb`

- Analysis of behavioral variables

#### `2a_letter_decoding.ipynb`

- Implements population decoding models
- Trains classifiers to predict letters
- Evaluates decoding accuracy

#### `2b_singleNeuron_response_box.ipynb`

- Analyzes single neuron responses

#### `2c_RT_INOUT.ipynb`

- Analyzes the relation between response time and letter decoding accuracy

#### `3a_heldout_letter_decoding.ipynb`

- Examines letter decoding when holding out one subject

#### `3b_response_time_change.ipynb`

- Examines reaction times across conditions

#### `3c_area_letter_decoding.ipynb`

- Compares decoding across brain areas

#### `3d_size_letter_decoding.ipynb`

- Compares decoding across set sizes

#### `3e_inout_letter_decoding.ipynb`

- Compares decoding across trial types

#### `glm.R`

- Fits a mixed-effect generalized linear model for the reaction time

---

## Results

The analysis produces:

- Decoding accuracy results
- Summary results of behavioral variables
- Mixed-effect linear model results for the reaction time
- Area- and condition-specific comparisons

Results are visualized in `Figures/` and archived in `Results.zip`.

Typical runs complete in minutes for notebooks 0a, 1a, 1b, 2b, 2c, 3b and for glm.R.

Notebooks 2a, 3a, 3c, 3d and 3e require multiple decoding runs. Each decoding run completes in around 2 minutes when using parallel processing (tested with 6 cores).

---

## Figures

The `Figures/` directory contains:

- Population decoding plots
- Single-neuron response plots
- Behavioral analysis figures
- Supplementary figures

Figures are generated automatically by notebooks.

---

## Reproducibility

To ensure reproducibility:

- Use fixed random seeds where provided
- Maintain consistent library versions
