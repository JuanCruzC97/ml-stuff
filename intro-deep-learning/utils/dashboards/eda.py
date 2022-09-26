import dash
from jupyter_dash import JupyterDash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc


def plot_3d_scatter(data, color=None):
    
    if color == "nada":
        color = None
    
    plot = px.scatter_3d(data_frame=data,
                         x="X1",
                         y="X2",
                         z="y",
                         color=color,
                         # paleta
                         # achicar barra lateral
                         # hoverinfo
                         color_continuous_scale="viridis",
                         height=600,
                         width=450,
                         template="plotly_white")
    
    plot.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    
    return plot

def plot_2d_scatter(data, variable, color, marginal=False):

    plot = px.scatter(data_frame=data,
                    x = variable,
                    y = "y",
                    color=color,
                    marginal_x=marginal,
                    marginal_y=marginal,
                    color_continuous_scale="viridis",
                    height=500,
                    width=500,
                    template ="plotly_white")
    
    plot.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                       coloraxis_showscale=False)
    
    return plot


    
    

def dataset_eda(datasets, y_pred=None):
    
    if y_pred is None:
        color_dw_options = [
             {"label": "Y", "value": "y"},
             {"label": "Ninguno", "value": "nada"}
        ]
    
    else:
        color_dw_options = [
             {"label": "Y", "value": "y"},
             {"label": "Ninguno", "value": "nada"},
             {"label": "Predicción", "value": y_pred}
        ]
        
    

    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.FLATLY], use_pages=False)

    app.layout = dbc.Container([
        
        dbc.Row([
                    
                    dbc.Col(
                        
                      dcc.Dropdown(id="data_dw", options=[{"label":"Train", "value":0}, {"label":"Test", "value":1}], value=0, clearable=False)  
                        
                    , width=3),
                    
                    dbc.Col(
                        
                      dcc.Dropdown(id="color_dw", options=color_dw_options, value="y", clearable=False)  
                        
                    , width=3)
                    
                ]),
        
        dbc.Row([
                        
            dbc.Col([
                
                dcc.Graph(id="3d_scatter", figure=plot_3d_scatter(datasets[0], color="y"))
                
            ], width = 6),
            
            dbc.Col([
                
                dcc.Graph(id="2d_scatter_x1", figure=plot_2d_scatter(datasets[0], variable="X1", color="y", marginal=False)),
                
                dcc.Graph(id="2d_scatter_x2", figure=plot_2d_scatter(datasets[0], variable="X2", color="y", marginal=False))
                    
            ], width=6)
                    
        ])], style={"margin":"10px"}
    
    )

    @app.callback(
        Output(component_id="3d_scatter", component_property="figure"),
        #Output(component_id="2d_scatter_x1", component_property="figure"),
        #Output(component_id="2d_scatter_x2", component_property="figure"),
        Input(component_id="data_dw", component_property="value"),
        Input(component_id="color_dw", component_property="value")
        
    )
    
    def update_plot(data_index, color_dw):
        return (plot_3d_scatter(datasets[data_index], color_dw))
        #return data_index


    app.run_server(mode='inline', port = 8090, dev_tools_ui=True, debug=True,
                dev_tools_hot_reload =True, threaded=True)