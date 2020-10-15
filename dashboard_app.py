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
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)

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

"""
<H1>  Title
<DCC> Features Importance Chart
<H4>  #1 Importance Feature Name
<DCC> #1 Feature slider
<H4>  #2 Importance Feature Name
<DCC> #2 Feature slider
<H4>  #2 Importance Feature Name
<DCC> #2 Feature slider
<H2>  Updated predictions
"""

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
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},

                    children=[

                        # Title display
                        html.H1(children="Bike Share Demand Dashboard"),

                        # Dash Graph Component calls the fig_features_importance parameters
                        dcc.Graph(figure=fig_features_importance),

                        # We display the most important feature's name
                        html.H4(children=slider_1_label),

                        # The Dash Slider is built according to Feature #1 ranges
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
                            marks={i: '{}'.format(i) for i in range(slider_3_min, slider_3_max+1)},
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

                        html.H4(children=slider_5_label),

                        dcc.Slider(
                            id='X5_slider',
                            min=slider_5_min,
                            max=slider_5_max,
                            step=1.0,
                            value=slider_5_mean,
                            marks={i: '{}'.format(i) for i in range(slider_5_min, slider_5_max+1)},
                        ),

                        # The predictin result will be displayed and updated here
                        html.H2(id="prediction_result"),

                    ])
# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"),
              Input("X4_slider","value"), Input("X5_slider","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3, X4, X5):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = np.array([X1,
                       X["atemp"].mean(),
                       X5,
                       X2,
                       X["month"].mean(),
                       X3,
                       X4]).reshape(1,-1)

    # Prediction is calculated based on the input_X array
    prediction = linear_m.predict(input_X)[0]
    prediction = np.expm1(prediction)

    # And retuned to the Output of the callback function
    return "Prediction: {}".format(round(prediction,1))

if __name__ == "__main__":
    app.run_server(debug=True)
