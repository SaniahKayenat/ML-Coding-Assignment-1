import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.H1("Predict what a car should be sold for!"),
        html.P(
            "The company uses this tool to get a quick estimate "
            "of what price a used car should have.",
            className="lead",
        ),
        html.Hr(),
        html.H4("Two models to choose from"),
        html.P(
            "There are two prediction pages in the menu above. They ask for the same "
            "information about the car, but they use different models underneath."
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Old model (A1)", className="card-title"),
                                html.P(
                                    "A Random Forest with 500 trees from "
                                    "Assignment 1.",
                                    className="card-text",
                                ),
                                dbc.Button("Use the old model", href="/predict",
                                           color="primary", outline=True),
                            ]
                        )
                    ),
                    md=6,
                    className="mb-3",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("New model (A2)", className="card-title"),
                                html.P(
                                    "A linear regression model from scratch in Assignment 2, "
                                    "trained with stochastic gradient descent on polynomial "
                                    "features.",
                                    className="card-text",
                                ),
                                dbc.Button("Use the new model", href="/predict-new",
                                           color="success"),
                            ]
                        )
                    ),
                    md=6,
                    className="mb-3",
                ),
            ]
        ),
        html.Hr(),
        html.H4("How it works"),
        html.Ol(
            [
                html.Li(
                    "Pick one of the two prediction pages above and fill in what you "
                    "know about the car."
                ),
                html.Li(
                    "If you don't know some technical specs, just leave those fields blank. They "
                    "will be estimated for you automatically."
                ),
                html.Li(
                    "You can leave all fields empty if you want and it will still predict a price for you using imputation methods used during training."
                    "But ideally, you should fill as many fields as you can."
                ),
                html.Li(
                    "Click \u201cPredict Price\u201d to get an instant estimated "
                    "selling price."
                ),
            ]
        ),
        html.P(
            "Both models were trained on the same 6000+ real used-car listings.",
            className="text-muted",
        ),
    ],
    fluid=True,
    className="py-4",
)
