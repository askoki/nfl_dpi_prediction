NFL DPI Prediction
==============================

## 1. Running scripts

### Linux
- Create .env from .env_example 
- Run ./postactivate.sh to add your directory to python path

### Windows
Procedure on windows is a little bit different
- Load root directory to python path with:

`conda develop .`

### Requirements
Running processing and training scripts requires at least 16 GB of RAM.


## 2. Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── nfl_requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         conda list --export > nfl_requirements.txt
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── create_v3_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── helpers <- Processing helpers
    │   │   ├── plot <- Processing for plotting and visualisation
    │   │   └── v3 <- Dataset creation steps
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   ├── ann
    │   │   │   ├── models.py <- Model architecture
    │   │   │   ├── predict_ann.py <- Model evaluation
    │   │   │   └── train.py <- Model training
    │   │   ├── gru
    │   │   ├── lstm
    │   │   ├── mlstm
    │   │   └──  grid_search_models.py <- Run grid search on all models
    │   │   
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├──generate_grid_search_report.py <- Generate 'per model' statistics for trained models
    │       └──accumulate_grid_search_report.py <- Gather 'per model' grid search reports into single file
    │
    └── settings.py            <- Settings file with constants and configuration variables.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
