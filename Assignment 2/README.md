# A2: Predicting Car Price

AT82.03 Machine Learning — Assignment 2 (st126956)

Continues from [Assignment 1](https://github.com/SaniahKayenat/ML-Coding-Assignment-1). The data cleaning and EDA are
the same; the modelling is replaced by a linear regression written from scratch, compared with MLflow, and deployed
with the old model side by side.

**Live site:** http://web-st126956.ml.brain.cs.ait.ac.th/ (also on https)

## Folder structure

```
.
├── A2_car_price_prediction.ipynb   # Task 2: experiment + report
├── regression_models.py            # Task 1: the model classes
├── a2_cv_results.csv               # all 144 cross-validation runs
├── Cars.csv
├── model/
│   └── car_price_model_a2.pkl      # the chosen model (preprocessor + weights)
├── Screenshots/                    # MLflow screenshots used in the report
└── app/
    ├── .Dockerfile
    ├── docker-compose.yaml         # local build / testing
    ├── docker-compose_forDeploy.yaml
    └── code/
        ├── main.py                 # navbar + page container
        ├── utils.py                # save() / load()
        ├── requirements.txt
        ├── test_app_callbacks.py
        ├── models/
        │   ├── car_price_model.pkl      # A1 Random Forest
        │   └── car_price_model_a2.pkl   # A2 from-scratch model
        └── pages/
            ├── home.py
            ├── predict.py          # old model page
            └── predict_new.py      # new model page
```

## Task 1 — Implementation

`regression_models.py` holds the model code, based on the `LinearRegression` class from
`03 - Regularization.ipynb`. I added:

- `r2()` alongside `mse()`.
- **Xavier initialisation** as an alternative to zeros (`weight_init='zeros' | 'xavier'`), drawing weights from
  U[-1/√m, 1/√m] where m is the number of inputs.
- **Momentum** (`momentum` in [0,1)), using the standard velocity form `step = lr*grad + momentum*prev_step`.
- `plot_feature_importance()`, a bar chart of the coefficients sorted by magnitude.

For the Task 2 experiment I also added stochastic GD (`method='sto'`), a `NoRegularization` penalty so "normal"
regression exists, a `PolynomialRegression` subclass, per-fold r², a divergence guard for settings that overflow,
and `fit_final()` to retrain the chosen settings on the whole training set. The intercept is not regularised, since
the target is log price (mean ≈ 12.9) and shrinking the bias towards zero would drag every prediction down.

## Task 2 — Experiment

Everything is in `A2_car_price_prediction.ipynb`, with the written report in the last section.

You can view them with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Task 3 — Deployment

The Dash site now has two prediction pages, picked from the navbar or the cards on the home page:

- **Old model (A1)** at `/predict` is the Random Forest pipeline.
- **New model (A2)** at `/predict-new` is the from-scratch model.

Both forms allow every field to be left blank; missing values are imputed the same way as during training.

### Run locally

```bash
cd app/code
pip install -r requirements.txt
python main.py          # http://127.0.0.1:8050
DEBUG=1 python main.py  # with the Dash debugger
```

With Docker, from `app/`:

```bash
docker compose up --build   # http://localhost:8050
```

### Deploy to ml-brain

```bash
# local: build and push
docker compose up --build
docker tag car-price-predictor:latest psyduckait/car-price-predictor:latest
docker push psyduckait/car-price-predictor:latest

# server
ssh -i <key> -J st126956@bazooka.cs.ait.ac.th st126956@ml.brain.cs.ait.ac.th
cd ~/st126956
docker compose up -d
```

The deploy compose file is `app/docker-compose_forDeploy.yaml`

The router rule is written out explicitly rather than relying on Traefik's default subdomain naming, and
`traefik.docker.network=web` is set because the container sits on two networks.

```

```
