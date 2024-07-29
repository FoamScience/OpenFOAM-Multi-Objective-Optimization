#!/usr/bin/env python3

""" Perform Insitu visualization of important metrics

This script defines functions to create Pandas dataframes out
of current optimization (or parameter variation) state.

The current principle is to chug all info through CSV files.
Seperate visualization toolkits (rg. Dash) can pick up these files.

IO operations are not intensive, this should be enough
"""

import hydra, logging, os
import subprocess as sb
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from scipy import stats

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots

from foambo.core import process_input_command

log = logging.getLogger(__name__)

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA],
        external_scripts=[{
            'src': "https://kit.fontawesome.com/f62ef3c170.js",
            'crossorigin': "anonymous"
            }])

def choose_icon(st):
    return "fa-check" if st == "COMPLETED" else "fa-square-xmark" \
        if st == "FAILED" else "fa-circle-exclamation"

def choose_color(st):
    return "4e9a06" if st == "COMPLETED" else "a40000" \
        if st == "FAILED" else "fcaf3e"


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config.yaml")
def dash_main(cfg : DictConfig):
    app.title = cfg.problem.name
    GRAPH_HEIGHT = 300 if "graph_height" not in cfg.visualize.keys() else cfg.visualize.graph_height
    @app.callback(Output('live-update-graph', 'figure'),
                  Input('interval-component', 'n_intervals'))
    def update_graph(fig):
        data = pd.DataFrame()
        try:
            data = pd.read_csv(f"{cfg.problem.name}_report.csv")
        except:
            log.warn("Could not visualize current state")
            return fig
        nrows = len(cfg.problem.objectives.keys())
        fig = make_subplots(
            rows=nrows, cols=1,
            specs=[[{'type': 'xy'}] for _ in range(nrows)],
        )
        i=1
        for key, _ in cfg.problem.objectives.items():
            obj = data[key].dropna()
            df = pd.DataFrame(obj[(np.abs(stats.zscore(obj)) < cfg.visualize.zscore_bar)], columns=[key])
            ifig = px.scatter(df, x=df.index, y=key, hover_name=key, hover_data=df.columns)
            fig.add_trace(
                ifig['data'][0],
                row=i, col=1
            )
            fig['layout']['xaxis{}'.format(i)]['title']='trial_index'
            fig['layout']['yaxis{}'.format(i)]['title']=key
            i += 1
        return fig

    @app.callback(Output('images', 'children'),
                  Input('interval-component', 'n_intervals'))
    def update_images(children):
        data = pd.DataFrame()
        try:
            data = pd.read_csv(f"{cfg.problem.name}_report.csv")
        except:
            log.warn("Could not visualize current state")
            return []
        df = data.tail(cfg.visualize.n_figures)
        figure_uris = []
        for _, row in df.iterrows():
            case = OmegaConf.create({"name": cfg.meta.clone_destination+row["casename"]})
            image_uri = sb.check_output(list(process_input_command(cfg.visualize.figure_generator,
                case)), cwd=case.name, stderr=sb.PIPE)
            figure_uris.append({ **row.to_dict(),
                "image": image_uri.decode("utf-8").strip(' ').replace('\"', '').replace('\\n', '')})
        return [
            html.Div(style={'width': f'{100/cfg.visualize.n_figures}%', 'float': 'left', 'position': 'relative'},
            children=[
                html.Img(src=uri["image"] if not ("null" in uri["image"] or uri["image"] == "") else "https://placehold.co/600x400/png",
                        width='95%', style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                html.I(className=f'fa-solid {choose_icon(uri["trial_status"])}',
                       style={"color": f'#{choose_color(uri["trial_status"])}', "position": "absolute",
                    "top": "0", "right": "0"}),
                html.Div(children=[
                    html.P(children=elm)
                    for elm in OmegaConf.to_yaml(OmegaConf.create(uri)).splitlines()
                    ])
            ])
        for uri in figure_uris ]

    updates = dcc.Interval(
            id='interval-component',
            interval=float(cfg.visualize.update_interval)*1000, # in milliseconds
            n_intervals=0,
    )

    app.layout = html.Div(children=[
        updates,
        html.H1(children=f'Bayesian Optimization run for {cfg.problem.name}',
                style={'text-align':'center', 'padding':'20px'}),
        dcc.Markdown(
f"""
## Optimization metrics

Objectives to minimize where "lower is better":
{[it[0] for it in cfg.problem.objectives.items() if it[1].minimize and it[1].lower_is_better]}

""", style={'padding':'20px'}),
        dcc.Graph(id='live-update-graph', figure={"layout": {"height": len(cfg.problem.objectives.keys())*GRAPH_HEIGHT}}),
        dcc.Markdown(
f"""
## Insight into latest {cfg.visualize.n_figures} trials

> Screenshot are uploaded to [IMGBB](https://imgbb.com/) by default, so make sure `IMGBB_API_KEY` is set

This section is controlled through `visualize` configuration set from the configuration file.
Here are the most important settings to look at:
- `figure_generator` points to a shell script to generate a screenshot of current simulation state.
- `update_interval` (in seconds) controls how often to **run the generator**. The generator can always
  cache the uploaded image URL as to not mount a DDOS attack on IMGBB :)
- `zscore_bar` culls penalized trials (outliers) from the metrics plots (assumed to be failing).
""", style={'padding':'20px'}),
        html.Div(id='images', style={'padding':'20px'}),
        html.Div(children=[
            dcc.Markdown(
f"""
## Your configuration

This visualization was executed with the following configuration file:

```yaml
{OmegaConf.to_yaml(cfg)}
```
""",  style={'white-space': 'pre-wrap'})],
            style = {'padding': '20px', 'margin': '10px', 'white-space': 'pre-wrap'}
        )
    ])
    app.run_server(debug=False, port=int(cfg.visualize.port), host=cfg.visualize.host)

if __name__ == '__main__':
    dash_main()
