# Predicting movie revenue from pre-release features

A machine-learning course project (Python, scikit-learn). It is a supervised regression task: predict a film's box-office revenue, on a log scale, using only information that is available before the film is released.

Notebook: [`predicting_movie_revenue.ipynb`](predicting_movie_revenue.ipynb)

## Data

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) on Kaggle (`movies_metadata.csv` and `credits.csv`), merged into 45,463 records. About 84% of the records have a revenue of zero, which in most cases means the revenue was not reported. The model is therefore trained on released films with a valid release date and strictly positive revenue, after further filtering of missing or noisy budget and runtime values. The final modeling set has 7,389 films and 74 features.

## Method

- **Target:** `log(1 + revenue)`.
- **Features:** genre indicators, budget, runtime, release year and encoded categorical fields, plus time-aware track-record features. For the five top-billed actors and for the director, writer and producer, the notebook computes the median revenue and the number of their earlier films. It uses only films released before the film in question, to avoid target leakage.
- **Split:** temporal, not random. Films released before 2015 form the training set (6,683 films) and films from 2015 onward form the test set (706 films). The split year was not tuned.
- **Model selection:** grid search with 3-fold cross-validation on the training period only, over Ridge regression, random forest and histogram gradient boosting. The test set is used only for the final evaluation.
- **Seeds:** fixed (`random_state=42`).

## Results

Cross-validated mean absolute error (MAE) on the training period, in log space:

| Model | CV MAE |
|---|---|
| HistGradientBoosting (selected; `learning_rate=0.03`, `max_depth=6`, `max_iter=300`) | 1.364 |
| Random forest | 1.391 |
| Ridge | 1.718 |

Final evaluation of the selected model on the held-out test set (706 films, log space): **MAE 1.3894**, **RMSE 1.9175**. Since exp(1.3894) ≈ 4.01, a typical prediction is off by a factor of about 4 on the original revenue scale. The notebook therefore interprets predictions as ranges, not point estimates.

## Limitations

The notebook discusses these in Section 9:

- Roughly 84% of the records have no usable revenue, so the modeling set favors better-documented, more visible films.
- Only pre-release information is used. Marketing spend, distribution, release window, competing releases and franchise effects are not modeled.
- Cast and crew features are approximations of star power and track record (top five billed actors; median past revenue and number of past films).
- With a typical error factor of about 4, the model is a coarse forecaster, not a precise estimator.

## Running the notebook

The notebook was developed in Google Colab. Its first cells install the dependencies (`kagglehub`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `joblib`, `tqdm-joblib`) and download the dataset with `kagglehub` (`rounakbanik/the-movies-dataset`). Run it from top to bottom.
