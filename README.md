# Predicting Restaurant Failure: An NLP-Driven Risk Assessment from Yelp Reviews

A big data analytics project that tests whether the language in customer reviews can signal that a restaurant is likely to close. The pipeline is built with **PySpark** on the **Yelp Open Dataset** and combines TF-IDF text features, a sentiment model, and date features to classify restaurants as open or closed.

**Course:** AID304 Big Data Analytics, International University of Sarajevo (IUS), Spring 2026
**Authors:** Asja Bašović, Hatidža Imamović

## Results

| Model | AUC-ROC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9566** | **0.8991** | 0.8991 | 0.8991 | 0.8991 |
| SVM (LinearSVC) | 0.9544 | 0.8872 | 0.8872 | 0.8873 | 0.8872 |
| Random Forest | 0.8302 | 0.7430 | 0.7421 | 0.7460 | 0.7430 |

Metrics are computed per review on a held-out test set. The train/test split is done by business, so no restaurant appears in both sets. Logistic Regression was the final model. The linear models clearly outperform Random Forest, which suggests the high-dimensional TF-IDF features are close to linearly separable.

## Pipeline

1. **Data acquisition and joining.** Three Yelp JSON files (`business`, `review`, `checkin`) are joined on `business_id` with Spark SQL. Only restaurants are kept (filtered by category), and unneeded columns are dropped.
2. **Class balancing.** Open restaurants greatly outnumber closed ones, so they are downsampled at the business level (hash-based), followed by stratified sampling. The result is a roughly balanced dataset of about 190,000 reviews and 14 columns.
3. **EDA.** Reviews per restaurant, businesses per state and per city, rating distribution, and open/closed counts.
4. **Preprocessing.** Two text-cleaning pipelines: a stricter one for TF-IDF (letters only) and a lighter one for sentiment (keeps numbers, `!`, `?`). URLs and emails are removed, duplicates are dropped, and dates are parsed.
5. **Feature engineering.**
   - TF-IDF on unigrams and bigrams (vocabulary of 5,000 each, minimum document frequency 50). "not" is kept out of the stopword list so bigrams can capture negation.
   - A sentiment model (Naive Bayes on TF-IDF, trained to predict a binary version of the review star rating) whose output probability is used as a feature.
   - Cyclical sine/cosine encoding of month and day for both the review date and the last check-in date, plus the year.
   - Standardization of the assembled feature vector.
6. **Modeling.** Logistic Regression, LinearSVC, and Random Forest (20 trees, max depth 10), chosen after a short literature review.
7. **Evaluation.** AUC-ROC, accuracy, F1, weighted precision, and weighted recall, plus an error analysis of closed restaurants that the model predicted would stay open, and accuracy by review length.

## Repository structure

```
AID304_Project.ipynb    Full pipeline: data prep, EDA, features, models, evaluation
final_dataset.csv       100 random rows of the final dataset (sample only)
README.md
```

## How to run

The notebook was developed in **Google Colab** with data stored on Google Drive.

1. Download the Yelp Open Dataset from <https://business.yelp.com/data/resources/open-dataset/> and place these files in your Drive (`MyDrive`):
   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_review.json`
   - `yelp_academic_dataset_checkin.json`
2. Open `AID304_Project.ipynb` in Colab and run the cells in order. The first cells install PySpark and OpenJDK 17, and mount Drive.
3. If you run it locally instead of in Colab, install Java 17, then `pip install pyspark xgboost nltk pandas numpy matplotlib seaborn`, and change the data paths in the loading cells.

The Spark session is configured for about 8 GB of driver and executor memory, since the full review file is several GB.

## Tech stack

Python, PySpark (Spark SQL, Spark ML), NLTK, pandas, NumPy, matplotlib, seaborn, Google Colab

## Data

The full dataset is not included. The Yelp Open Dataset must be downloaded from Yelp under its own terms. `final_dataset.csv` contains only a 100-row random sample for illustration.

## Limitations and future work

- Some restaurants close for reasons unrelated to service quality, such as relocation or economic conditions. The model cannot detect these cases, and the reviews of such restaurants are often positive.
- Accuracy is measured per review, not per restaurant. Aggregating predictions per restaurant would be a natural next step.
- Ideas for improvement: a sentiment trend feature over time, hyperparameter tuning and cross-validation, additional models such as XGBoost, and running on a distributed cluster for larger-scale experiments.
