# Spaceship Titanic — End-to-End ML Workflow

A complete machine learning project based on Kaggle's Spaceship Titanic competition.

The main goal of this project was not just to train a model and get a good score. I wanted to build a complete ML workflow, from the raw data all the way to model deployment with an API.

## Project overview

The task is to predict whether a passenger was transported to another dimension after the Spaceship Titanic accident.

The project follows this workflow:

Raw data → preprocessing → feature engineering → model comparison → hyperparameter tuning → model interpretation → error analysis → Kaggle submission → API

## Dataset

The dataset comes from the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic).

The main features include information about:

- Home planet
- CryoSleep
- Cabin
- Destination
- Age
- VIP status
- Spending in different areas of the ship

The target variable is `Transported`.

## Feature engineering

Before training the models, I created some additional features from the original data.

Some examples are:

- `TotalSpend`: total amount spent by a passenger
- `HasSpentMoney`: whether the passenger spent any money
- `GroupSize`: size of the passenger's group
- `IsAlone`: whether the passenger was travelling alone
- `Deck`: extracted from the `Cabin` column
- `CabinSide`: extracted from the `Cabin` column

I also removed features such as `PassengerId` and `Name` when they were not useful for the model.

## Models

I compared three different models:

- Logistic Regression
- Random Forest
- CatBoost

All models were evaluated using cross-validation so that the results were not based on a single train/validation split.

After comparing them, CatBoost performed best, so I used it as the final model.

## Hyperparameter tuning

I used `RandomizedSearchCV` to search for better CatBoost parameters.

The final selected parameters included:

- `learning_rate = 0.1`
- `iterations = 500`
- `depth = 7`

The tuned model achieved around **0.806 accuracy in cross-validation**.

## Model explainability with SHAP

I used SHAP to understand what the CatBoost model was actually using to make its predictions.

The most important features included variables such as:

- Spa spending
- VRDeck spending
- RoomService spending
- HomePlanet
- FoodCourt
- CryoSleep
- Cabin side
- Total spending

I also used SHAP waterfall plots to look at individual predictions and see which features pushed a prediction towards `Transported=True` or `False`.

## Error analysis

After training the final model, I looked at where the model was making mistakes instead of only looking at the overall accuracy.

One interesting pattern was that the model performed much worse for passengers whose `HomePlanet` was Earth:

| HomePlanet | Accuracy | Error rate |
| --- | ---: | ---: |
| Earth | 75.99% | 24.01% |
| Europa | 95.17% | 4.83% |
| Mars | 89.82% | 10.18% |

I then investigated the Earth passengers in more detail.

One of the clearest patterns was `CryoSleep`:

- Earth + `CryoSleep=False` → 20.3% error
- Earth + `CryoSleep=True` → 32.6% error

I also found that incorrectly classified Earth passengers tended to have lower total spending than correctly classified passengers.

This does not prove that these features cause the errors, but they give a useful idea of where the model struggles.

## Kaggle submission

After training the final model, I generated predictions for the Kaggle test set and created a `submission.csv`.

The submission was successfully uploaded to Kaggle.

My final Kaggle score was:

**0.80500**

This was slightly lower than the accuracy obtained on the training data, which is expected since Kaggle evaluates the predictions on unseen data.

## API

I also turned the trained model into a small REST API using FastAPI.

The API exposes a `/predict` endpoint where a passenger can be sent as JSON.

For example:

```json
{
  "HomePlanet": "Earth",
  "CryoSleep": false,
  "Destination": "TRAPPIST-1e",
  "Age": 39,
  "VIP": false,
  "RoomService": 0,
  "FoodCourt": 0,
  "ShoppingMall": 0,
  "Spa": 0,
  "VRDeck": 0,
  "Cabin": "B/0/P"
}
```

The API returns the model prediction:

```json
{
  "Transported": true
}
```

The model and preprocessing pipeline are loaded from the saved `.joblib` file, so the API does not retrain the model when receiving a request.

## Project structure

```text
professional-ml-workflow/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_...
│   ├── 02_...
│   ├── 03_...
│   ├── 04_...
│   ├── 05_error_analysis.ipynb
│   └── 06_kaggle_submission.ipynb
│
├── src/
│   ├── features/
│   ├── models/
│   └── api.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

