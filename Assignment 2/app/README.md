# This repository contans all the code and report for coding assignment 1 (Task 1 to 3).

## Task 1 and 2

The notebook named `car_price_prediction.ipynb` contains the processes for data loading, EDA, preprocessing and analysis. It also compares multiple
ML models to find the best one for the dataset and saves the model for future use. Inference is also done using two sets of dummy data, where once all input features were provided and once some inputs were missing. In the missing inputs, values were filled using imputation method used in the project. A PDF generated from the notebook is also provided and a two paragraph analysis of the project is provided at the bottom of the PDF and notebook.

## Folder structure (Task 3)

```
app/
├── .Dockerfile
├── docker-compose.yaml             # for local dev / testing
├── docker-compose_forDeploy.yaml
├── README.md
└── code/
    ├── main.py                     # entry point: builds the navbar + page container
    ├── utils.py                    # save()/load() helpers
    ├── requirements.txt
    ├── test_app_callbacks.py       # pytest tests for the prediction logic
    ├── models/
    │   └── car_price_model.pkl     # trained pipeline exported from the notebook
    └── pages/
        ├── home.py                 # landing page with instructions
        └── predict.py              # the form, callback, and predict_price()
```

## Run locally (no Docker)

```bash
cd app/code
pip install -r requirements.txt
python main.py
```

Then open http://127.0.0.1:8050 (Dash's default) in your browser.

## Run the tests

```bash
cd app/code
pytest test_app_callbacks.py -v
```

These call `predict.predict_price()` directly (no server needed) and check
that: predictions come back as positive numbers, missing optional fields
don't break anything, newer cars price higher than older ones (all else
equal), and more km driven doesn't price higher than less km driven.

## Run with Docker

From the `app/` folder:

```bash
docker compose up --build
```

Then open http://localhost:8050. Stop with `docker compose down`.

## Deploying to the course server

`docker-compose_forDeploy.yaml` is a template based on the reporsitory https://github.com/chaklam-silpasuwanchai/Python-for-Machine-Learning/tree/main/Appendix/Appendix%20-%20Dash%20Plotly . But maybe due to changes in the domain or network
name, it gives an error of external network not found when

```bash
docker compose -f docker-compose_forDeploy.yaml up --build -d
```

is run.

## How the prediction works

1. The user fills in the required fields on the **Predict Price**
   page.
2. Ideally, brands, year, km_driven, fuel, seller type, transmission,
   ownership history fields should be required. But as per the instructions
   on the assignment, users are allowd to leave any of all fields empty.
   Whichever field is empty will be filled automatically using the imputation method applied in the model training.
3. On clicking **Predict Price**, the callback in `pages/predict.py` calls
   `predict_price(...)`, which builds a one-row DataFrame with the same
   column names used during training and passes it through the loaded
   pipeline (`models/car_price_model.pkl`).
4. The pipeline predicts on the log-price scale; `predict_price()` applies
   `np.exp` to convert back to the original selling price before it's
   displayed.
