"""
this is an app for the dashboard running a linear regression
model to estimate the demand of bike rentals for Bike Share in Washington D.C.
"""
import pandas as pd
from scipy import stats
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

DATA = pd.read_csv('train_dataset.csv')

def create_timefs(df):
    """
    create datetime features from datetime column
    """
    df = df.copy()
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    return df

df = create_timefs(DATA)
df = df.drop(['datetime'], axis=1)
z = np.abs(stats.zscore(df))
df = df[(z < 3).all(axis=1)]
feature_list = ['temp', 'atemp', 'workingday', 'hour', 'month', 'weather', 'humidity']
X = (df[feature_list])
y = np.log1p(df["count"])

linear_m = make_pipeline(PolynomialFeatures(degree=8), Ridge(max_iter=3000, alpha=0.0001, normalize=True))
linear_m.fit(X,y)

col_names = ['temp', 'atemp', 'workingday', 'hour', 'month', 'weather', 'humidity']
df_feature_importances = pd.DataFrame([10, 10, 7, 6, 2, 5, 3], columns=["Importance"],index=col_names)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)
# We create a Features Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )

# We record the name, min, mean and max of the three most important features
slider_1_label = "Temperature"
slider_1_min = 0
slider_1_mean = 20
slider_1_max = 50

slider_2_label = "Hour"
slider_2_min = 0
slider_2_mean = 11
slider_2_max = 23

slider_3_label = "Weather"
slider_3_min = 1
slider_3_mean = 1
slider_3_max = 4

slider_4_label = "Humidity"
slider_4_min = 0
slider_4_mean = 60
slider_4_max = 100

slider_5_label = "Workingday"
slider_5_min = 0
slider_5_mean = 0
slider_5_max = 1

slider_6_label = "Month"
slider_6_min = 1
slider_6_mean = 6
slider_6_max = 12

options_months = [
    {'label': 'January', 'value': 1},
    {'label': 'February', 'value': 2},
    {'label': 'March', 'value': 3},
    {'label': 'April', 'value': 4},
    {'label': 'May', 'value': 5},
    {'label': 'June', 'value': 6},
    {'label': 'July', 'value': 7},
    {'label': 'August', 'value': 8},
    {'label': 'September', 'value': 9},
    {'label': 'October', 'value': 10},
    {'label': 'November', 'value': 11},
    {'label': 'December', 'value': 12}]


###############################################################################

app = dash.Dash()

# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    <H4> Feature #3 name
#    Slider to update Feature #3 value
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output

# We apply basic HTML formatting to the layout
app.layout = html.Div([

    # First Row
    html.Div([
        # Image and Input container left
        html.Div([
            html.Img(id="bike image",
                     height="180px",
                     src="assets/bike_flipped.jpg",
                     style={"border-radius": "20px"}),

            # Title display
            html.H1(children="How many bikes will be rented?"),

            # We display the most important feature's name
            html.H4(children="Prediction:", style={"fontSize": "25px"}),
            html.H2(id="prediction_result", style={"fontSize": "60px"}),
        ], className="pretty-container three columns"),

        # Title and main-sliders container right
        html.Div([
            html.Div([
                html.H1('Dashboard Capital Bikeshare',
                        style={"textAlign": "center",
                               "display":"flex",
                               "alignItems":"center",
                               "justifyContent": "center"})
            ], className="pretty-container"),

            # The Dash Slider is built according to Feature #1 ranges
            html.Div([

                html.H4(children=slider_1_label),

                dcc.Slider(
                    id='X1_slider',
                    min=slider_1_min,
                    max=slider_1_max,
                    step=1.0,
                    value=slider_1_mean,
                    marks={i: '{}°'.format(i) for i in range(slider_1_min, slider_1_max+1, 5)}
                    ),

                html.H4(children=slider_2_label),

                dcc.Slider(
                    id='X2_slider',
                    min=slider_2_min,
                    max=slider_2_max,
                    step=1.0,
                    value=slider_2_mean,
                    marks={i: '{}h'.format(i) for i in range(slider_2_min, slider_2_max+1)}
                ),

                html.H4(children=slider_3_label),

                dcc.Slider(
                    id='X3_slider',
                    min=slider_3_min,
                    max=slider_3_max,
                    step=1.0,
                    value=slider_3_mean,
                    marks={1: 'Clear', 2: 'Cloudy', 3: 'Light Rain', 4: 'Heavy Rain'},
                ),

                html.H4(children=slider_4_label),

                dcc.Slider(
                    id='X4_slider',
                    min=slider_4_min,
                    max=slider_4_max,
                    step=5.0,
                    value=slider_4_mean,
                    marks={i: '{}%'.format(i) for i in range(slider_4_min, slider_4_max+1, 5)},
                ),

                html.H4(children=slider_6_label),

                dcc.Dropdown(
                    id='X6_slider',
                    className="input-line",
                    style={"flex-grow":"3",},
                    options=options_months,
                    value=1),

                html.H4(children=slider_5_label),

                dcc.RadioItems(
                    id='X5_slider',
                    options=[{"label": "Yes", 'value':1},
                            {"label": "No", 'value':0}],
                    value=1)],
                    className="pretty-container")

        ], className="basic-container-column twelve columns"),

    ], className="basic-container"),


    # Second Row
    html.Div([
        html.H3("Influence of weather-conditions",
                style={"textAlign": "center",
                       "fontSize": "20px",
                       "fontWeight": "normal"})

    ], className="pretty-container"),

    # Third Row
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_features_importance)])
    ], className="pretty-container twelve columns")

], className="general")


###############################################################################

###############################################################################
# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"),
              Input("X4_slider","value"), Input("X5_slider","value"), Input("X6_slider","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3, X4, X5, X6):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = np.array([X1,
                       X["atemp"].mean(),
                       X5,
                       X2,
                       X6,
                       X3,
                       X4]).reshape(1,-1)

    # Prediction is calculated based on the input_X array
    prediction = linear_m.predict(input_X)[0]
    prediction = np.expm1(prediction)

    # And retuned to the Output of the callback function
    return "{}".format(int(prediction))

if __name__ == "__main__":
    app.run_server(debug=True)
