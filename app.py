import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import os
import nltk.data
import numpy as np

app = dash.Dash(__name__)
server = app.server
app.title = 'Sentiment Analysis of Dante\'s Divine Comedy'
#settings color scheme
#some colors for the DIVs are set in the css code inside assets
#graph colors mainly need to be edited on the graphs
colors = {
    'div-background':'#262B3D',
    'plot-background':'#262B3D',
    'navbar-backgorund':'#262B3D',
    'navbar-text':'#E2EFFA',
}


##read data##
df_inferno = pd.read_csv('data/infernoderivative.csv')
df_purgatorio = pd.read_csv('data/purgatorioderivative.csv')
df_paradiso = pd.read_csv('data/paradisoderivative.csv')
df_urls = pd.read_csv('data/URL_Directory_Gutenberg.csv')
############

selectionMap = {
    'Inferno': df_inferno,
    'Purgatorio': df_purgatorio,
    'Paradiso': df_paradiso,
}

#helper function for canto callback
def get_text(section, canto):
    root = 'data'
    fileName = "Canto_" + str(canto) + ".txt"
    path = os.path.join(root, section, fileName)
    data = ""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(path, 'r') as file:
        data = file.read()
    return tokenizer.tokenize(data)

#return a graph of a single section
def update_graph_single(section, df):

    sentiment_color = 'rgb(99, 110, 250)'
    first_derivative_color = 'rgb(239, 85, 59)'
    second_derivative_color = 'rgb(0, 204, 150)'
    seperator_color = "#2F3591"

    annotation_size = 18
    title_size = 30
    axis_size = 22
    tick_size = 13

    sentiment_width = 6
    derivative_width = 1.4

    zeroline_width = 0.8
    # HOVER CODE per Section
    hover_code = ["{}: <br>Canto #{}".format(section, x) for x in range(1, len(df['Canto']))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[1,len(df)], y=[0, 0],
            mode='lines',
            hoverinfo="skip",
            showlegend=False,
            name='0 line - dashed',
            line=dict(width=zeroline_width,
                    color = seperator_color,
                    dash = "dash"))
    )
    fig.add_trace(
        go.Scatter(x=df['Canto'], y=df['Sentiment'],
                mode='lines+markers',
                hovertext=hover_code,
                hoverinfo="text",
                name='Sentiment Polarity',
                line=dict(width=sentiment_width, color = sentiment_color))
        )
    fig.add_trace(
        go.Scatter(x=df['Canto'], y=df['FirstDerivative'],
            mode='lines',
            hoverinfo='skip',
            name='First Derivative',
            line=dict(width=derivative_width, color = first_derivative_color))
    )
    fig.add_trace(
        go.Scatter(x=df['Canto'], y=df['SecondDerivative'],
            hoverinfo='skip',
            mode='lines', name='Second Derivative',
            line=dict(width=derivative_width, color = second_derivative_color))
    )

    fig.update_layout(
        margin=go.layout.Margin(
            b=150
        ),
        font = dict(
            color="#FAF0CA"
        ),
        xaxis=dict(
            title = "Canto",
            showline=True,
            showgrid=False,
            showticklabels=True,
            zeroline = False,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=tick_size,
                color="#FAF0CA",
            ),
            titlefont = dict(
                size = axis_size
            )
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
        showlegend=True,
        plot_bgcolor=colors['plot-background'],
        paper_bgcolor=colors['plot-background'],
        title = dict(
            text = section + ": Sentiment per Canto",
            x = 0.5,
            font=dict(
                size=title_size,
            ),
        ),
        #margin = {'l':5, 'r':5, 'b':5, 't':15, 'pad': 0, 'autoexpand': True},

    )

    return fig

#returns graph with all sections
def update_graph_all():
    # Create traces
    sentiment_color = 'rgb(99, 110, 250)'
    first_derivative_color = 'rgb(239, 85, 59)'
    second_derivative_color = 'rgb(0, 204, 150)'
    seperator_color = "rgb(245,245,245)"

    annotation_size = 18
    title_size = 30
    axis_size = 22
    tick_size = 13

    sentiment_width = 4
    derivative_width = 1.4
    zeroline_width = 0.8

    # HOVER CODE
    inferno_hover = ["Inferno: <br>Canto #{}".format(x) for x in range(1, 35)]
    purgatorio_hover = ["Purgatorio: <br>Canto #{}".format(x) for x in range(1, 34)]
    paradiso_hover = ["Paradiso: <br>Canto #{}".format(x) for x in range(1, 34)]


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=[1,99], y=[0, 0],
                        mode='lines',
                        hoverinfo="skip",
                        showlegend=False,
                        name='0 line - dashed',
                        line=dict(width=zeroline_width,
                                color = seperator_color,
                                dash = "dash"))
    )

    fig.add_trace(
        go.Scatter(x=df_inferno['Canto'], y=df_inferno['Sentiment'],
                        mode='lines+markers',
                        hovertext=inferno_hover,
                        hoverinfo="text",
                        name='Sentiment Polarity - Inferno',
                        line=dict(width=sentiment_width,
                                color = sentiment_color))
    )

    fig.add_trace(
        go.Scatter(x=df_inferno['Canto'], y=df_inferno['FirstDerivative'],
                        mode='lines',
                        name='First Derivative',
                        legendgroup = "inferno_derivatives",
                        hoverinfo='skip',
                        line=dict(width=derivative_width, color = first_derivative_color))
    )

    fig.add_trace(
        go.Scatter(x=df_inferno['Canto'], y=df_inferno['SecondDerivative'],
                        mode='lines', name='Second Derivative',
                        hoverinfo='skip',
                        legendgroup = "inferno_derivatives",
                        line=dict(width=derivative_width, color = second_derivative_color))
    )

    # start purgatorio
    fig.add_trace(
        go.Scatter(x=[34, 34], y=[-1, 1],
                        mode='lines',
                        hoverinfo='skip',
                        name='Purgatorio Splitter',
                        showlegend=False,
                        line=dict(color=seperator_color, width=1.2,
                                dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(x=[x + 33 for x in df_purgatorio['Canto']], y=df_purgatorio['Sentiment'],
                        mode='lines+markers',
                        name='Sentiment Polarity - Purgatorio',
                        hovertext=purgatorio_hover,
                        hoverinfo="text",
                        showlegend=True,
                        line=dict(width=sentiment_width,
                                color = sentiment_color))
    )
    fig.add_trace(
        go.Scatter(x=[x + 33 for x in df_purgatorio['Canto']], y=df_purgatorio['FirstDerivative'],
                        mode='lines',
                        hoverinfo='skip',
                        name='First Derivative',
                        legendgroup = "purgatorio_derivatives",
                        showlegend=True,
                        line=dict(width=derivative_width, color = first_derivative_color))
    )
    fig.add_trace(
        go.Scatter(x=[x + 33 for x in df_purgatorio['Canto']], y=df_purgatorio['SecondDerivative'],
                        mode='lines', name='Second Derivative',
                        legendgroup = "purgatorio_derivatives",
                        showlegend=True,
                        hoverinfo='skip',
                        line=dict(width=derivative_width, color = second_derivative_color))
    )

    # start paradiso
    fig.add_trace(
        go.Scatter(x=[66, 66], y=[-1, 1],
                        mode='lines',
                        name='Paradiso Splitter',
                        showlegend=False,
                        hoverinfo='skip',
                        line=dict(color=seperator_color, width=1.2,
                                dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(x=[x + 65 for x in df_paradiso['Canto']], y=df_paradiso['Sentiment'],
                        mode='lines+markers',
                        name='Sentiment Polarity - Paradiso',
                        hovertext=paradiso_hover,
                        hoverinfo="text",
                        showlegend=True,
                        line=dict(width=sentiment_width,
                                color = sentiment_color))
    )
    fig.add_trace(
        go.Scatter(x=[x + 65 for x in df_paradiso['Canto']], y=df_paradiso['FirstDerivative'],
                        mode='lines',
                        name='First Derivative',
                        hoverinfo='skip',
                        legendgroup = "paradiso_derivatives",
                        showlegend=True,
                        line=dict(width=derivative_width, color = first_derivative_color))
    )
    fig.add_trace(
        go.Scatter(x=[x + 65 for x in df_paradiso['Canto']], y=df_paradiso['SecondDerivative'],
                        mode='lines', name='Second Derivative',
                        legendgroup = "paradiso_derivatives",
                        showlegend=True,
                        hoverinfo='skip',
                        line=dict(width=derivative_width, color = second_derivative_color))
    )


    # TICK CODE
    tickvals_full = np.linspace(1,101,100)
    tickvals_full = tickvals_full[::5]
    ticktext_inferno = ["{}".format(x) for x in range(1, 35)]
    ticktext_purgatorio = ["{}".format(x) for x in range(1, 34)]
    ticktext_paradiso = ["{}".format(x) for x in range(1, 34)]
    ticktext_full = ticktext_inferno + ticktext_purgatorio + ticktext_paradiso
    ticktext_full = ticktext_full[::5]


    fig.update_layout(
        font = dict(
            color="#FAF0CA"
        ),
        plot_bgcolor=colors['plot-background'],
        paper_bgcolor=colors['plot-background'],
        xaxis=dict(
            title = "Canto",
            showline=True,
            showgrid=False,
            showticklabels=True,
            zeroline = False,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            rangeslider=dict(
                visible=True
            ),
            tickangle=-45,
            ticks='outside',
            tickvals = tickvals_full,
            ticktext = ticktext_full,
            tickfont=dict(
                family='Arial',
                size=tick_size,
                color="#FAF0CA",
            ),
            titlefont = dict(
                size = axis_size
            )
            
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            range = [-0.7,0.7],
            tickfont=dict(
                family='Arial',
                size=tick_size,
                color="#FAF0CA",
            )
        ),
        showlegend=True,
        title = dict(
            text = "Average Sentiment per Section and Canto",
            x = 0.5,
            font=dict(
                size=title_size,
            ),
        ),
        annotations=[
            go.layout.Annotation(
                x=6,
                y=0.55,
                xref="x",
                yref="y",
                text="Inferno",
                showarrow=False,
                arrowhead=7,
                ax=0,
                ay=-40,
                font=dict(
                    size=annotation_size,
                    color="#FAF0CA"
                ),
            ),
            go.layout.Annotation(
                x=42,
                y=0.55,
                xref="x",
                yref="y",
                text="Purgatorio",
                showarrow=False,
                arrowhead=7,
                ax=0,
                ay=-40,
                font=dict(
                    size=annotation_size,
                    color="#FAF0CA"
                )
            ),
            go.layout.Annotation(
                x=74.0,
                y=0.55,
                xref="x",
                yref="y",
                text="Paradiso",
                showarrow=False,
                arrowhead=7,
                ax=0,
                ay=-40,
                font=dict(
                    size=annotation_size,
                    color="#FAF0CA"
                ),
            )
        ]
    )

    return fig

#returns a graph with average sentiment per section
def update_graph_average():
    
    avg_sentiment_inferno = df_inferno['Sentiment'].mean()
    avg_sentiment_purgatorio = df_purgatorio['Sentiment'].mean()
    avg_sentiment_paradiso = df_paradiso['Sentiment'].mean()

    avg_first_inferno = df_inferno['FirstDerivative'].mean()
    avg_first_purgatorio = df_purgatorio['FirstDerivative'].mean()
    avg_first_paradiso = df_paradiso['FirstDerivative'].mean()

    avg_second_inferno = df_inferno['SecondDerivative'].mean()
    avg_second_purgatorio = df_purgatorio['SecondDerivative'].mean()
    avg_second_paradiso = df_paradiso['SecondDerivative'].mean()

    sections = ['Inferno', 'Purgatorio', 'Paradiso']
    mean_sent = [avg_sentiment_inferno, avg_sentiment_purgatorio, avg_sentiment_paradiso]
    mean_first_deriv = [avg_first_inferno, avg_first_purgatorio, avg_first_paradiso]
    mean_second_deriv = [avg_second_inferno, avg_second_purgatorio, avg_second_paradiso]

    float_formatter = "{:.2f}".format

    fig = go.Figure(
        data=[
            go.Bar(name='Mean Sentiment', x=sections, y=mean_sent, text=[ "{:0.3f}".format(x) for x in mean_sent],
                textposition='auto'),
            go.Bar(name='Mean First Derivative', x=sections, y=mean_first_deriv, text=[ "{:0.3f}".format(x) for x in mean_first_deriv],
                textposition='auto'),
            go.Bar(name='Mean Second Derivative', x=sections, y=mean_second_deriv, text=[ "{:0.3f}".format(x) for x in mean_second_deriv],
                textposition='auto')
    ])

    fig.update_layout(
        font = dict(
            color="#FAF0CA"
        ),
        plot_bgcolor=colors['plot-background'],
        paper_bgcolor=colors['plot-background'],
        barmode='group',
        xaxis=dict(
            tickfont=dict(
                family='Arial',
                size=15,
                color="#FAF0CA",
            ),
        ),
        showlegend=True,
        title = dict(
            text = "Average Sentiment per Section and Canto",
            x = 0.5,
        )
    )
    return fig



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
                html.Img(src='assets/ForensX.png', height='30', width='39',
                         style={'top': '10', 'margin': '10px'})
            ],
            href='/Portal'
        ),
        html.H2("Sentiment Analysis of Dante's Divine Comedy"),
        html.A(
            id='gh-link',
            children = ["View on GitHub"],
            href = "https://github.com/forensx/DanteSentimentAnalysis",
            style = {'color': "white", 'border':"solid 1px white"}
        ),
        html.Img(
            src = "https://raw.githubusercontent.com/plotly/dash-sample-apps/master/apps/dashr-uber-rasterizer/assets/GitHub-Mark-64px.png"
        )

    ]
)
#####################

tabs = html.Div(
    dcc.Tabs(id = 'circos-control-tabs', value='Canto', children=[
        dcc.Tab(
            label='Canto',
            value='Canto',
            children=html.Div(
                id='canto-tab',
                children=[
                    html.H4('Select a canto by clicking on the points on the graph!')
                ],
            ),
        ),
        dcc.Tab(
            label='About',
            value='About',
            children=html.Div(
                id='control-tab',
                children=[
                    html.H4('Lit HL - English Project', style={'font-size':'24pt', 'font-wight':'200', 'letter-spacing':'1px'}),
                    html.Div(
                        style={'padding':'5px', 'fontSize':'15px'},
                        children=[
                            dcc.Markdown('''
                                In this work, we investigate the use of modern sentiment analysis algorithms in the analysis of Dante’s Divine Comedy. 
                                We utilize lexicon-trained sentiment analysis per canto through the three sections of Dante’s Inferno, Purgatorio, and Paradiso. 
                                For the investigation, we utilize the Python 3.6 scripting language, the Pandas module for data processing, and the TextBlob module for text processing and sentiment polarity calculations. 
                                We discuss average sentiment along with average rates of changes and points of inflections in the story’s chronology.


                                **Aniket Pant, Viraj Kacker, Cole McKee, Lonnie Webb**
                                ''')
                        ]
                    )
                ]
            )
        ),

    ])
)
options = html.Div([
    tabs
], className='item-a col-md-5')

app.layout = html.Div(children=[
    header,
    html.Div(
        children=[
            options,
            html.Div(
                children=[
                    #TODO create the options bar right here
                    dcc.Dropdown(
                        id='section-dropdown',
                        options=[
                            {'label': 'All Sections', 'value': 'all'},
                            {'label': 'Inferno', 'value': 'Inferno'},
                            {'label': 'Purgatorio', 'value': 'Purgatorio'},
                            {'label': 'Paradiso', 'value': 'Paradiso'},
                            
                            {'label': 'Average Sentiment', 'value': 'avg'},
                        ],
                        value='all'
                    ),
                    dcc.Graph(
                        id='dante-graph',
                        figure=update_graph_all(),
                        style={
                            'height':'80vh',
                            'color':'#FAFOCA',
                        },
                        className = 'item-b'
                    )
                ],
                style={
                    'background': "#262B3D",
                },
                className = 'item-b col-md-7',
            )
        ],
        className = 'container'
    )
])


#callback to update the graph
@app.callback(
    Output('dante-graph', 'figure'),
    [Input('section-dropdown', 'value')]
)
def update_graph(value):
    if value == 'all':
        return update_graph_all()
    elif value  == 'avg':
        return update_graph_average()
    df = selectionMap[value]
    return update_graph_single(value, df)

#callback to update canto
@app.callback(
    Output('canto-tab', 'children'),
    [Input('dante-graph', 'clickData'),
     Input('section-dropdown', 'value')]
)
def update_canto(selection, section):
    canto = ""
    cantoNumber = 0
    if selection is None:
        return html.P("Select a canto by clicking on the points on the graph!")
    elif section =='all':
        hovertext = selection['points'][0]['hovertext']
        cantoNumber = hovertext[hovertext.find('#')+1:]
        section = hovertext[:hovertext.find(':')]
        data = get_text(section, cantoNumber)
        first = " ".join(sentence for sentence in data[:1])
        second = " ".join(data[-1:])
        canto = first + " [...] " + second

    else:
        cantoNumber = (selection['points'][0]['x'])
        data = get_text(section, cantoNumber)
        first = " ".join(sentence for sentence in data[:1])
        second = " ".join(data[-1:])
        canto = first + " [...] " + second

    sentiment = round(selection['points'][0]['y'], 4)
    urls = df_urls[df_urls['Section'] == section]['Link'].tolist()
    url = urls[int(cantoNumber)-1]
    return html.Div(
        children=[
            html.H3(section + ' - Canto #' + str(cantoNumber), style = {"fontSize": 18}),
            html.P("Sentiment for Canto = " + str(sentiment), style={'fontSize': 16}),
            html.A(url, href=url, style={'fontSize':13,'color':'hotpink'}),
            html.P(canto)
        ]
    )

if __name__ == "__main__":
    app.run_server(threaded = True, port=8080, debug=True)