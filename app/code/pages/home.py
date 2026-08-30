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
        html.H4("How it works"),
        html.Ol(
            [
                html.Li(
                    "Go to the \u201cPredict Price\u201d page above and fill in what you "
                    "know about the car."
                ),
                html.Li(
                    "If you don't know some technical specs, just leave those fields blank. They "
                    "will be estimated for you automatically."
                ),
                html.Li(
                    "You can leave all fields empty if you want and it will still predict a price for you."
                    "But ideally, you should fill as many fields as you can."
                ),
                html.Li(
                    "Click \u201cPredict Price\u201d to get an instant estimated "
                    "selling price."
                ),
            ]
        ),
        html.P(
            "Under the hood, this uses a Random Forest model trained on 6000+ "
            "real used-car listings.",
            className="text-muted",
        ),
    ],
    fluid=True,
    className="py-4",
)
