from dash import dcc
from dash import html
import dash
import plotly
import plotly.graph_objs as go
import pandas as pd
import warnings
from flask import Flask
from dash.dependencies import Input, Output, State
import os
import xgboost
from sklearn.metrics import roc_curve, auc, precision_recall_curve


warnings.filterwarnings("ignore")

server = Flask(__name__)

app = dash.Dash(f'Model Comparison',server=server)              ## Create Dash App


#------------ Reading files from directory and creating lists for features, labels and models ----------#

all_files = os.listdir()
feature_files = []
label_files = []
model_files = []

for file in all_files:
    if 'parquet' in file:                       ## Features and labels can only be parquet files
        if 'features' in file:                  ## Features file must have "features" and Labels file must have "labels" in the name
            feature_files.append(file)
        elif 'labels' in file:
            label_files.append(file)
    elif 'pickle' in file or 'json' in file:    ## Model can be either pickle or json
        model_files.append(file)



#----------------------- Initial App Layout ------------------------#


app.layout = html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','justifyContent':'center','width':'100%'}, children=[

        # App title

        html.Br(),
        html.H1(f'Model Comparison', style = {'textAlign':'center','color':'white','white-space':'pre','fontSize':50}),
        html.Br(),
        html.Br(),
        html.Br(),

        # Div which contains a label and dropdown for data set (features)

        html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'20%',
        'justifyContent':'center','paddingLeft':'40%'}, children=[

             html.Label('Select dataset (features)',
             style = {'textAlign':'center','color':'white','white-space':'pre','fontSize':40, 'margin': '20px'}),
             
             html.Br(),
             html.Br(),

             dcc.Dropdown(
                id = 'dropdown_features',
                options = [{'label': i, 'value': i} for i in feature_files],        # populate options from files
                multi=False,
                searchable=False,               
                clearable=True,
                optionHeight=70,
                style = {
                'height': '50px',
                'borderWidth': '2px',
                'borderStyle': 'solid',
                'borderRadius': '5px',
                'textAlign': 'center',
                'borderColor': 'grey',
                'backgroundColor':'white',
                'color':'black',
                'fontSize':30,
                },

            )

        ]),

        html.Br(),
        html.Br(),
        html.Br(),

        # Div which contains a label and dropdown for data set (labels)

        html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'20%',
        'justifyContent':'center','paddingLeft':'40%'}, children=[

             html.Label('Select dataset (labels)',
             style = {'textAlign':'center','color':'white','white-space':'pre','fontSize':40, 'margin': '20px'}),
             
             html.Br(),
             html.Br(),

             dcc.Dropdown(
                id = 'dropdown_labels',
                options = [{'label': i, 'value': i} for i in label_files],          # populate options from files
                multi=False,
                searchable=False,
                clearable=True,
                optionHeight=70,
                style = {
                'height': '50px',
                'borderWidth': '2px',
                'borderStyle': 'solid',
                'borderRadius': '5px',
                'textAlign': 'center',
                'borderColor': 'grey',
                'backgroundColor':'white',
                'color':'black',
                'fontSize':30,
                },

            )

        ]),
        
        html.Br(),
        html.Br(),
        html.Br(),

        # Outer Div which contains 2 inner divs each cotaining a label and dropdown for both models

        html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','display':'flex'}, children=[

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'20%',
            'justifyContent':'center','paddingLeft':'15%'}, children=[

                html.Label('Select first model',
                style = {'textAlign':'center','color':'white','white-space':'pre','fontSize':40, 'margin': '20px'}),
                
                html.Br(),
                html.Br(),

                dcc.Dropdown(
                    id = 'dropdown_model_1',
                    options = [{'label': i, 'value': i} for i in model_files],              # populate options from files
                    multi=False,
                    searchable=False,
                    clearable=True,
                    optionHeight=70,
                    style = {
                    'height': '50px',
                    'borderWidth': '2px',
                    'borderStyle': 'solid',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'borderColor': 'grey',
                    'backgroundColor':'white',
                    'color':'black',
                    'fontSize':30,
                    },

                )

            ]),

            # Dropdown for second model

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'20%',
            'justifyContent':'center','paddingLeft':'30%'}, children=[

                html.Label('Select second model',
                style = {'textAlign':'center','color':'white','white-space':'pre','fontSize':40, 'margin': '20px'}),
                
                html.Br(),
                html.Br(),

                dcc.Dropdown(
                    id = 'dropdown_model_2',
                    options = [{'label': i, 'value': i} for i in model_files],                  # populate options from files
                    multi=False,
                    searchable=False,
                    clearable=True,
                    optionHeight=70,
                    style = {
                    'height': '50px',
                    'borderWidth': '2px',
                    'borderStyle': 'solid',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'borderColor': 'grey',
                    'backgroundColor':'white',
                    'color':'black',
                    'fontSize':30,
                    },

                )

            ])
        
        ]),

        
        html.Br(),
        html.Br(),
        html.Br(),

        # Div for Generate graphs button

        html.Div(
                    html.Button('Generate Graphs', 
                        id='submit', 
                        n_clicks=0, 
                        style = {'fontSize':30,'height':'50px','width':'300px','backgroundColor':'lightgreen'}),
                    style = {'display':'flex','justifyContent':'center'}
                ),
        
        html.Br(),
        html.Br(),

        # Div to output graphs when generated - gets updated when button clicked

        html.Div(id='output-data-upload',style = {
            'color':'black','height': '100vh'}),
        
    ])


#------------ Function to generate predictions once files are chosen ------------#

def generate_predictions(features, labels, model):

    pred_labels = model.predict(features)
    pred_probs = model.predict_proba(features)
    test_pred = pd.DataFrame()
    test_pred['new_label'] = pred_labels
    test_pred['probability'] = [prob[1] for prob in pred_probs]
    test_pred['true_label'] = labels['label'].values
    
    return test_pred


#--------------- Function to generate PR Curve and return a dcc.graph object -------------#


def layout_prec_rec(test_pred,model):

    precision, recall, thresholds = precision_recall_curve(test_pred['true_label'], test_pred['probability'])
    pr_auc_test = auc(recall, precision)

    x = recall
    y = precision
    model_name = model.split('.')[0]

    # lineData is a scatterplot for P/R

    lineData = plotly.graph_objs.Scatter(
                            x=x,
                            y=y,
                            name=f"Precision Recall Curve - {model_name}",
                            customdata=thresholds,                                  # To include threshold values in hoverdata
                            mode="lines",
                            line = dict(color='cyan', width=4),
                            hovertemplate='<extra></extra>'+'<br>'.join([           # Custom hoverdata for each point
                                'Precision: %{y:.2f}',
                                'Recall: %{x:.2f}',
                                'Threshold: %{customdata:.3f}'])
                            
                        )

    title_string = f'PR Curve - {model_name} <br>PR AUC: {round(pr_auc_test,3)}<br>'     # Title of the plot


    # Layout of the plot

    layout = go.Layout(
                    title=dict(text=title_string, font=dict(size=26,color='white')),
                    xaxis={'title':'Recall',
                            'titlefont' : dict(
                                    family = 'Arial, sans-serif',
                                    size = 24,
                                    color = 'white'
                                    ),
                            'zeroline':True,
                            'tickfont' : dict(
                                    size = 20,
                                    color='white'
                                    ),
                            'showgrid':False,
                            'range':[0,1.01]

                        },
                    yaxis = {'title':'Precision',
                            'titlefont' : dict(
                                    family = 'Arial, sans-serif',
                                    size = 24,
                                    color = 'white'
                                    ),
                            'zeroline':True,
                            'tickfont' : dict(
                                    size = 20,
                                    color='white'
                                    ),
                            'showgrid':False,
                            'range':[0,1.01]
                            },
                    hoverlabel=dict(
                                bgcolor="white",
                                font_size=26
                            ),                            
                    paper_bgcolor = '#111111',
                    plot_bgcolor = '#111111',
                    showlegend=False,
                    )


    # Dotted line from (0,0) to (1.01,1.01)

    dotted_line = go.Scatter(x=[0,1.01],y=[0,1.01],line=dict(color='grey', width=4,dash='dash'))

    return dcc.Graph(figure = go.Figure(data = [lineData,dotted_line],layout = layout),style={'height':'90%'})        


#--------------- Function to generate ROC Curve and return a dcc.graph object -------------#


def layout_roc(test_pred,model):

    fpr, tpr, thresholds = roc_curve(test_pred['true_label'], test_pred['probability'])
    roc_auc_test = auc(fpr, tpr)

    x = fpr
    y = tpr
    model_name = model.split('.')[0]

    # lineData is a scatterplot for FPR/TPR

    lineData = plotly.graph_objs.Scatter(
                            x=x,
                            y=y,
                            name=f"ROC Curve - {model_name}",
                            mode="lines",
                            textfont=dict(size=30),
                            line = dict(color='cyan', width=4),
                            customdata=thresholds,                                      # To include threshold values in hoverdata
                            hovertemplate='<extra></extra>'+'<br>'.join([               # Custom template for hoverdata
                                'FPR: %{x:.2f}',
                                'TPR: %{y:.2f}',
                                'Threshold: %{customdata:.3f}'])
                        )
                        
    title_string = f'ROC Curve - {model_name} <br>ROC AUC: {round(roc_auc_test,3)}<br>'  # Title of plot

    # Layout for plot

    layout = go.Layout(
                    title=dict(text=title_string, font=dict(size=26,color='white')),
                    xaxis={'title':'False Positive Rate',
                            'titlefont' : dict(
                                    family = 'Arial, sans-serif',
                                    size = 24,
                                    color = 'white'
                                    ),
                            'zeroline':True,
                            'tickfont' : dict(
                                    size = 20,
                                    color='white'
                                    ),
                            'showgrid':False,
                            'range':[0,1.01]

                        },
                    yaxis = {'title':'True Positive Rate',
                            'titlefont' : dict(
                                    family = 'Arial, sans-serif',
                                    size = 24,
                                    color = 'white'
                                    ),
                            'zeroline':True,
                            'tickfont' : dict(
                                    size = 20,
                                    color='white'
                                    ),
                            'showgrid':False,
                            'range':[0,1.01]
                            },
                    hoverlabel=dict(
                                bgcolor="white",
                                font_size=26
                            ), 
                    paper_bgcolor = '#111111',
                    plot_bgcolor = '#111111',
                    showlegend=False)


    # Dotted line from (0,0) to (1.01,1.01)

    dotted_line = go.Scatter(x=[0,1.01],y=[0,1.01],line=dict(color='grey', width=4,dash='dash'))


    return dcc.Graph(figure = go.Figure(data = [lineData,dotted_line],layout = layout),style={'height':'90%'})


#--------------- Function to generate Feature Importance Graph and return a dcc.graph object -------------#


def layout_feature_imp(model,model_name):

    feature_imp = model.get_booster().get_score()       ## Get feature importance dict

    keys = list(feature_imp.keys())                     ## Names of features are not stored, so default names (f1-f33)
    values = list(feature_imp.values())                 ## Importance of each feature

    model_name = model_name.split('.')[0]

    # Sort features by importance
    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)

    x = data.score
    y = data.index

    # barData is a horizontal bar plot

    barData = go.Bar(
        x = x,
        y = y,
        marker_color = 'cyan',
        orientation='h',
        hovertemplate='<extra></extra>'+'<br>'.join([           # Custom template for hoverdata
                'Feature: %{y}',
                'Importance: %{x}'])
    )
    

    # Layout for graph
    layout = go.Layout(
                    title=dict(text=f'Feature Importance - {model_name}', font=dict(size=26,color='white')),
                    xaxis={'title':'Importance',
                            'titlefont' : dict(
                                    family = 'Arial, sans-serif',
                                    size = 24,
                                    color = 'white'
                                    ),
                            'zeroline':True,
                            'tickfont' : dict(
                                    size = 20,
                                    color='white'
                                    ),
                            'showgrid':False

                        },
                    yaxis = {'title':'Feature',
                            'titlefont' : dict(
                                    family = 'Arial, sans-serif',
                                    size = 24,
                                    color = 'white'
                                    ),
                            'zeroline':True,
                            'tickfont' : dict(
                                    size = 16,
                                    color='white'
                                    ),
                            'showgrid':False,
                            'type':'category',
                            'dtick':1                       # Ensures all y-axis ticks (feature names) are displayed
                            },
                    hoverlabel=dict(
                                bgcolor="white",
                                font_size=26
                            ), 
                    paper_bgcolor = '#111111',
                    plot_bgcolor = '#111111')


    return dcc.Graph(figure = go.Figure(data = barData,layout = layout),style={'height':'90%'})


#---------------- Main Function which is called when generate button is pressed --------------#


def generate_output(features,labels,model_1,model_2):

    ## Check if selected files have the right format
    try:
        sel_features = pd.read_parquet(features)
        sel_labels = pd.read_parquet(labels)

        first_model = xgboost.XGBClassifier()
        first_model.load_model(model_1) 
        second_model = xgboost.XGBClassifier()
        second_model.load_model(model_2)

        first_preds = generate_predictions(sel_features,sel_labels,first_model)
        second_preds = generate_predictions(sel_features,sel_labels,second_model)

    ## If error reading files or generating predictions, output an error and return
    except:
        output_data = html.Div(style={'backgroundColor': '#111111'}, children=[
            html.Br(),
            html.Br(),
            html.H1(f'CHECK INPUT FILES - BAD FORMATTING', style = {'textAlign':'center','color':'red','white-space':'pre','fontSize':50})
        ])
        
        return output_data

    # If no error then define the layout of the output


    #-------------------- Graph Data layout --------------------#
    
    
    output_data = html.Div(style={'backgroundColor': '#111111','height':'150vh'}, children=[

        ## Div for Precision-Recall Curve - includes 2 divs, one for each model

        html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','display':'flex','height':'30%'},children=[

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'40%',
            'justifyContent':'center','paddingLeft':'5%'},children=[

                layout_prec_rec(first_preds,model_1)
            ]),

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'40%',
            'justifyContent':'center','paddingLeft':'10%'},children=[

                layout_prec_rec(second_preds,model_2)
            ]),
        ]),

        ## Div for ROC Curve - includes 2 divs, one for each model

        html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','display':'flex','height':'30%'},children=[

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'40%',
            'justifyContent':'center','paddingLeft':'5%'},children=[

                layout_roc(first_preds,model_1)
            ]),

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'40%',
            'justifyContent':'center','paddingLeft':'10%'},children=[

                layout_roc(second_preds,model_2)
            ]),
        ]),

        ## Div for Feature Importance curve - includes 2 divs, one for each model

        html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','display':'flex','height':'40%'},children=[

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'40%',
            'justifyContent':'center','paddingLeft':'5%'},children=[

                layout_feature_imp(first_model,model_1)
            ]),

            html.Div(style={'backgroundColor': '#111111','horizontalAlign':'center','textAlign':'center','width':'40%',
            'justifyContent':'center','paddingLeft':'10%'},children=[

                layout_feature_imp(second_model,model_2)
            ]),
        ]),

    ])

    return output_data


#------------------------ CALLBACK FUNCTIONS ---------------------#

# Callback to populate labels dropdown based on feature selection

@app.callback(Output('dropdown_labels','value'),
              Input('dropdown_features','value'))
def update_label_field(features):
    if features:                                                                    ## If feature selected

        selected_feature = features.split('features')[0]                            ## Get starting string (everything before "features")
                                                                                    ## This only works if the starting string for the features and labels file is the same
        corr_labels = [file for file in label_files if selected_feature in file]    ## Get corresponding label file

        if len(corr_labels)==1:                                                     ## Only if there is 1 match, update value
            return corr_labels[0]   
        else:
            return


# Main callback to generate graphs

@app.callback(Output('output-data-upload', 'children'),
            [Input('dropdown_features', 'value'),
            Input('dropdown_labels', 'value'),
            Input('dropdown_model_1', 'value'),
            Input('dropdown_model_2', 'value'),
            Input('submit','n_clicks')])
def update_output(features,labels,model_1,model_2,clicks):

    ctx = dash.callback_context                                         ## Check what triggered callback

    if not ctx.triggered:
        button_id = 'No clicks yet'

    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'submit':                                           ## If submit button clicked, only then call function to render
        if features and labels and model_1 and model_2:                 ## If all values selected, call main function for output

            return generate_output(features,labels,model_1,model_2)

        else:                                                           ## If all values not selected, output default message

            output_data = html.Div(style={'backgroundColor': '#111111'}, children=[
                html.Br(),
                html.Br(),
                html.H1(f'PLEASE SELECT ALL OPTIONS', style = {'textAlign':'center','color':'red','white-space':'pre','fontSize':50})
            ])

            return output_data


server.run(debug = True, port=8080)                                      ## Run on specified port, debug=False if not making changes
                                                                         ## This will run on http://127.0.0.1:8080/
