# Household Energy Consumption & Air Conditioning Prediction

A machine learning project using the **U.S. Energy Information
Administration's Residential Energy Consumption Survey (RECS) 2009**
public dataset. It builds regression models to estimate household energy
consumption and classification models to predict whether a household has
air conditioning.

## Project Overview

Two prediction tasks are explored:

-   **Regression (`TOTALBTU`):** estimate household energy consumption.
-   **Classification (`AIRCOND`):** predict whether air conditioning is
    present (`1`) or absent (`0`).

The workflow covers data inspection, cleaning, exploratory data analysis
(EDA), model training, evaluation, visualization, and comparison.

## Models

  -----------------------------------------------------------------------
  Task                    Model                   Configuration
  ----------------------- ----------------------- -----------------------
  Regression              Multiple Linear         Scikit-learn
                          Regression              `LinearRegression`

  Regression              Polynomial Regression   Degree 2 +
                                                  `LinearRegression`

  Classification          Logistic Regression     Scikit-learn
                                                  `LogisticRegression`

  Classification          K-Nearest Neighbors     `n_neighbors=5`, with
                                                  `StandardScaler`
  -----------------------------------------------------------------------

## Dataset

-   **Source:** [RECS 2009 --- U.S. Energy Information
    Administration](https://www.eia.gov/consumption/residential/data/2009/)
-   **CSV expected by the notebook:** `recs2009_public.csv`
-   **Original dimensions:** 12,083 rows × 940 columns.
-   **After filtering invalid negative `BEDROOMS` values:** 11,868 rows.
-   **Eight input features:** `TOTSQFT_EN`, `TOTROOMS`, `BEDROOMS`,
    `HDD65`, `CDD65`, `NHSLDMEM`, `MONEYPY`, `YEARMADERANGE`.
-   **Targets:** `TOTALBTU` (regression) and `AIRCOND` (classification).

Outliers in `TOTALBTU` and `TOTSQFT_EN` were identified using the IQR
method and retained in the modelling data.

> The dataset is not included in this repository. Download the CSV from
> the official EIA source and place `recs2009_public.csv` in the same
> directory as the notebook.

## Tech Stack

-   Python and Jupyter Notebook
-   NumPy and Pandas
-   Matplotlib and Seaborn
-   Scikit-learn

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Create and activate a virtual environment (recommended)

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Add the dataset

Download `recs2009_public.csv` from the [official RECS 2009
page](https://www.eia.gov/consumption/residential/data/2009/) and put it
beside the notebook.

### 5. Run the notebook

``` bash
jupyter notebook
```

Open the project notebook (for example, `Household_Prediction.ipynb`)
and run its cells from top to bottom.

## Methodology

1.  Load the RECS 2009 CSV file.
2.  Inspect dimensions, data types, summary statistics, missing values,
    and duplicate rows.
3.  Remove records with invalid negative `BEDROOMS` values.
4.  Examine outliers and distributions using IQR calculations and plots.
5.  Perform EDA with histograms, bar charts, boxplots, scatter plots,
    and a correlation heatmap.
6.  Split the data into training and test sets using an 80:20 split and
    `random_state=42`.
7.  Train two regression models for `TOTALBTU`.
8.  Train two classification models for `AIRCOND`.
9.  Evaluate and compare the models.

## Results

### Regression --- `TOTALBTU`

  ----------------------------------------------------------------------------
  Model                   MAE                MSE           RMSE             R²
  ------------ -------------- ------------------ -------------- --------------
  Multiple          27,190.89   1,795,940,844.08      42,378.54         0.4547
  Linear                                                        
  Regression                                                    

  Polynomial        26,828.72   1,781,488,655.21      42,207.68         0.4591
  Regression                                                    
  (degree 2)                                                    
  ----------------------------------------------------------------------------

Polynomial Regression performs slightly better on all four metrics, but
the improvement over Multiple Linear Regression is small.

### Classification --- `AIRCOND`

  Model                   Accuracy   Recall (class 0)   Recall (class 1)   Macro F1
  --------------------- ---------- ------------------ ------------------ ----------
  Logistic Regression       82.18%               0.02               1.00       0.47
  KNN (K=5)                 82.43%               0.25               0.95       0.62

Confusion matrices (rows = actual class, columns = predicted class):

**Logistic Regression**

``` text
[[   7, 422],
 [   1, 1944]]
```

**KNN**

``` text
[[ 106, 323],
 [  94, 1851]]
```

KNN is the stronger classifier in this experiment because it identifies
more households in the minority class (`AIRCOND = 0`) and achieves a
higher macro F1-score. However, it still misses many households without
air conditioning.

## Key Observations

-   The regression models explain approximately 45--46% of the variation
    in `TOTALBTU` on the held-out test set.
-   Polynomial Regression provides only a modest improvement over
    Multiple Linear Regression.
-   The classification test set contains 429 records in class `0` and
    1,945 in class `1`, showing class imbalance.
-   Accuracy alone is not sufficient to assess the classifiers;
    class-wise recall and macro F1 reveal the difficulty of predicting
    the minority class.
-   KNN performs better than Logistic Regression on minority-class
    recall in this experiment.

## Limitations

-   Evaluation uses one train-test split; cross-validation was not
    performed.
-   Only eight features were used from the 940-column dataset.
-   Outliers were identified but retained.
-   The KNN value `K=5` was not tuned.
-   KNN uses scaled features, while Logistic Regression in the notebook
    does not.
-   Logistic Regression produced a convergence warning at the configured
    maximum of 100 iterations.
-   Class imbalance was not corrected using class weights or resampling.
-   These results are an academic modelling baseline, not a validated
    production prediction system.

## Future Improvements

-   Use cross-validation and repeated experiments for more reliable
    evaluation.
-   Tune hyperparameters, including the number of neighbors in KNN.
-   Scale features consistently where appropriate and revisit Logistic
    Regression convergence.
-   Evaluate class weights, resampling, and decision-threshold
    adjustments.
-   Explore additional RECS features and other regression and
    classification algorithms.
-   Investigate the effect of outlier treatment and RECS survey weights.
-   Consider building a prediction interface after further model
    validation.

## Suggested Repository Structure

``` text
.
├── README.md
├── Household_Prediction.ipynb
├── recs2009_public.csv       # Download separately; usually do not commit large raw data
└── requirements.txt          # Optional dependency list
```

Change the notebook filename in this example if your actual notebook has
a different name.

## References

1.  [U.S. Energy Information Administration --- RECS
    2009](https://www.eia.gov/consumption/residential/data/2009/)
2.  [Python Documentation](https://docs.python.org/3/)
3.  [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Disclaimer

This project is for educational and analytical purposes. The reported
metrics describe the specific train-test split and modelling setup used
in the notebook; they do not guarantee prediction accuracy on other
households or datasets.
