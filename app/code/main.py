"""
Car Price Prediction Task 3 (Deployment)

Entry point for the multi-page Dash app.

Run locally with:   python3 main.py
Run in Docker with: docker compose up --build   (more in /README.md)
"""

import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

# incorporate a Bootstrap theme, same as the course reference project
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # exposed in case you want to run this behind gunicorn later

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Predict Price", href="/predict")),
    ],
    brand="Chaky's Car Price Predictor",
    brand_href="/",
    color="primary",
    dark=True,
)

app.layout = html.Div([
    navbar,
    dash.page_container,
])

# Run the app.
# (see docker-compose.yaml) otherwise Dash falls back to 127.0.0.1:8050.
if __name__ == "__main__":
    app.run(debug=True)
