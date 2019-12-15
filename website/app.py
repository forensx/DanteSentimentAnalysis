import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash(__name__)
server = app.server

##read data##
inferno_derivative = "/data/infernoderivative.csv"
inferno_sentiment = "/data/infernosentiment.csv"

paradiso_derivative = "/data/paradisoderivative.csv"
paradiso_sentiment = "/data/paradisosentiment.csv"

purgatorio_derivative = "/data/purgatorio_derivative.csv"
purgatorio_sentiment = "/data/purgatorio_sentiment.csv"
############


#TODO create working graph
update_graph():
    pass


########### Header #############
header = html.Div(
    id = 'app-page-header',
    style = {
        'width': "100%",
        'background': "#262B3D",
        'color': "#E2EFFA",
    },
    children = [
        html.A(
            id = 'dash-logo',
            children=[
                html.Img(src='https://raw.githubusercontent.com/plotly/dash-sample-apps/master/apps/dashr-uber-rasterizer/assets/plotly-dash-logo.png', height='36', width='180',
                         style={'top': '10', 'margin': '10px'})
            ],
            href='/Portal'
        ),
        html.H2("DANTE'S DIVINE COMEDY"),
        html.A(
            id='gh-link',
            children = ["View on GitHub"],
            href = "https://github.com/plotly/dash-sample-apps/tree/master/apps/dashr-uber-rasterizer",
            style = {'color': "white", 'border':"solid 1px white"}
        ),
        html.Img(
            src = "https://raw.githubusercontent.com/plotly/dash-sample-apps/master/apps/dashr-uber-rasterizer/assets/GitHub-Mark-64px.png"
        )

    ]
)
#####################

tabs = html.Div(
    dcc.Tabs(id = 'circos-control-tabs', value='what-is', children=[
        dcc.Tab(
            label='About',
            children=html.Div(
                id='control-tab',
                children=[
                    html.H4('OUR ENGLISH PROJECT', style={'font-size':'24pt', 'font-wight':'200', 'letter-spacing':'1px'}),
                    html.Div(
                        style={'padding':'5px'},
                        children=[
                            dcc.Markdown('''
                                This Dash app is a simple demonstration of the rasterizing capabilities of the _rasterly_ package.
                                The dataset consists of over 4.5 million observations, representing Uber rides taken in New York City in 2014.
                                In CSV format, the source data are over 165 MB in size. _rasterly_ is capable of processing datasets an order
                                of magnitude larger in similarly brisk fashion. The raster data required to produce the aggregation layers and color
                                gradients displayed here are computed efficiently enough to maintain the interactive feel of the application.
                                ''')
                        ]
                    )
                ]
            )
        ),
        dcc.Tab(
            label='Canto',
            children=html.Div(
                id='canto-tab',
                children=[
                    html.H4('Place Holder')
                ],
            ),
        ),

    ])
)
options = html.Div([
    tabs
], className='item-a')

app.layout = html.Div(children=[
    header,
    html.Div(
        children=[
            options,
            html.Div(
                children=[
                    #TODO create the options bar right here
                    dcc.Graph(
                        id='dante-graph',
                        figure=update_graph,
                        style={
                            'height':'88vh',
                        },
                        className = 'item-b'
                    )
                ],
                className = 'container',
            )
        ],
        className = 'container'
    )
])


#TODO implement callback to update canto based on graph

# @app.callback(
#     Output('canto-tab', 'children'),
#     [Input('graph', 'clickData')]
# )
# def update_canto(selection):
#     if selection is None:
#         canto = "Select a Canto"
#     else:
#         canto = "CANTO SELECTED"
#     return html.P(canto)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)