import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import os
import nltk.data
import numpy as np


def get_annotations(value):
    if value == "Inferno":
        annotations = ["Descent into Hell", "", "Descent into Hell", "", "Descent into Hell", "", "", "",
                       "", "", "", "", "", "", "", "", "Dante arrives at waterfall where Phlegethon<br>falls into eighth circle of Hell", "", "", "", "", "", "", "", "", "", "Sicilian Bull", "", "", "", "", "", ""]
    elif value == "Purgatorio":
        annotations = ["", "", "", "", "", "", "", "",
                       "", "", "", "", "", "", "", "", "", "", "", "", "Meeting Statius", "", "", "", "", "", "", "", "", "", "", ""]
    if value == "Paradiso":
        annotations = annotations = ["", "", "", "", "", "", "", "",
                                     "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]

    return annotations
