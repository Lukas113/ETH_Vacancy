import sys, warnings, datetime

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../database')

# from DBAccess import DBAccess
from DBAccess import DBAccess
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from data_wrangling import get_config, rename_person, time_interaction_table, group_by_hour, dayNameFromWeekday, add_min_max, filter_mal, group_norm_category
from datetime import date
import os
import yaml
from ipywidgets import interact, widgets
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, Markdown, Latex
import matplotlib.colors as colors



dba = DBAccess()
clrs = get_config()['color'] # four base colors for visualization


def save_plot(figure, title):
    """
    Saves plot in folder called plot_html
    :param figure: interactive plotly figure
    :param title: title for plot
    """
    # get current date
    today = str(date.today())

    # create folder if necessary
    if not os.path.exists(f'./plot_html/{today}'):
        os.makedirs(f'./plot_html/{today}')

    # get path to plot
    path = f'./plot_html/{today}/{title}.html'

    # save figure as html
    figure.write_html(path)


def p_rotary(prep_fun):
    """
    Plots a linechart for rotary elements in dataframe with interactive dropdown. Min-Max Values for sensors are displayed as well
    :param prep_fun: short for preparation function. Takes as input a function to preprocess the DataFrame (df)
    """
    clrs = get_config()['color']

    df_sens = dba.select('rotary_elements')

    df = prep_fun(df_sens)

    # define bounds
    ts = get_config()['turn_sensors']
    upper = {'Drehwand': ts['Maximum_Drehwand'], 'Drehschrank': ts['Maximum_Drehschrank'],\
             'LampeAussenwand': ts['Maximum_LampeAussenwand'], 'LampeDrehwand': ts['Maximum_LampeDrehwand']}
    lower = ts['Minimum']
    color = [clrs['Drehwand'][0], clrs['Drehschrank'][0], clrs['LampeAussenwand'][0], clrs['LampeDrehwand'][0], '#D55E00']  # last one is red for min/max marker

    fig = go.FigureWidget(layout={'height': 600})

    fig.add_trace(go.Scatter(
        x=[df['Drehwand'].sensor_timestamp.iloc[-1], df['Drehwand'].sensor_timestamp.iloc[0]],
        y=[lower, lower],
        mode="lines+text",
        name="Sensorminimum",
        text=["", "Minimalwert"],
        textposition="bottom right",
        showlegend=False,
        visible=False,
        line=dict(
            color=color[4],
            width=0.5
        ),
        textfont=dict(
            color=color[4])
    ))
    count = 0
    for val in df:
        data = df[val]
        fig.add_trace(
            go.Scatter(
                x=data.sensor_timestamp,
                y=data.sensor_numeric,
                name=val,
                line=dict(
                    color=color[count]
                ),

            )),

        fig.add_trace(go.Scatter(
            x=[df['Drehwand'].sensor_timestamp.iloc[-1], df['Drehwand'].sensor_timestamp.iloc[0]],
            y=[upper[val], upper[val]],
            mode="lines+text",
            name="Sensormaximim",
            text=["", "Maximalwert"],
            textposition="top right",
            showlegend=False,
            visible=False,
            line=dict(
                color=color[4],
                width=0.5
            ),
            textfont=dict(
                color=color[4])
        )),
        count += 1

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=list(
                [dict(label='All',
                      method='update',
                      args=[{'visible': [False, True, False, True, False, True, False, True, False]},
                            # the index of True aligns with the indices of plot traces
                            {'title': 'Alle beweglichen Sensoren',
                             'showlegend': True}], ),
                 dict(label='Drehschrank',
                      method='update',
                      args=[{'visible': [True, True, True, False, False, False, False, False, False]},
                            {'title': 'Drehschranksensor',
                             'showlegend': True}]),
                 dict(label='Drehwand',
                      method='update',
                      args=[{'visible': [True, False, False, True, True, False, False, False, False]},
                            {'title': 'Drehwandsensor',
                             'showlegend': True}]),
                 dict(label='LampeAussenwand',
                      method='update',
                      args=[{'visible': [True, False, False, False, False, True, True, False, False]},
                            {'title': 'LampeAussenwandsensor',
                             'showlegend': True}]),
                 dict(label='LampeDrehwand',
                      method='update',
                      args=[{'visible': [True, False, False, False, False, False, False, True, True]},
                            {'title': 'LampeDrehwandsensor',
                             'showlegend': True}]),
                 ]), direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1,
            xanchor="right",
            y=1,
            yanchor="top"
        )
        ])

    # Set title
    fig.update_layout(
        title_text="Drehelementwerte über die Zeit",
        xaxis_title="Zeit [Variabel]",
        yaxis_title="Sensorwert [Grad]",
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all",
                         label='Alle')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # save plot
    save_plot(figure=fig, title="Drehelementwerte über die Zeit")

    fig.show()


# grouped barplot
def grouped_barplot_week(data, y='count', y_axis_label = 'Total', title='Title'):
    """
    Plots a barplot for every day of the week
    :param data: pandas DataFrame containing a column 'sensor'
    :param y: Numeric column in Dataset for y axis (e.g. "count" or "mean")
    :param y_axis_label: Sets string in parenthesis of plot title and y_axis_label
    """
    g = sns.catplot(x='weekday', y=y, hue='sensor', data=data, kind="bar", height= 6,aspect=2.5, palette=[clr[0] for k, clr in clrs.items()])

    (g.set(title=f'{title}')
      .set_axis_labels('Wochentag', f'Anzahl Bewegungen ({y_axis_label})')
      .set_xticklabels(["Mo.", "Di.","Mi.","Do.","Fr.","Sa.", "So.", "Mo."])
      .despine(left=True)
      ._legend.set_title('Sensorname'))

    plt.show()

# amount of movement in dirty and clean dataset two seperate countplots
def turn_bar_diff(clean,dirty):
    """
    Plots two seperate barplots to visualize diffrence of dirty and clean dataset
    :param clean: cleaned log sensor dataset, flag=0
    :param dirty: dirty log sensor dataset, no flags cleared
    """
    # plotting
    sensors = ['Drehwand', 'Drehschrank', 'LampeDrehwand', 'LampeAussenwand']

    sns.set(rc={'figure.figsize':(15,6)})

    # countplot clean
    plt.subplot(1, 2, 1)
    fig  = sns.countplot(x= "sensor", order=sensors, data=clean,  orient='v')
    fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    fig.set(title='Totale Anzahl Bewegungen, Dataset clean', xlabel='Sensor',     ylabel='Anzahl Bewegungen')
    fig.axes.set_ylim(0, dirty[dirty['sensor'].isin(sensors)].sensor.value_counts()[0]+200)

    # countplot dirty
    plt.subplot(1, 2, 2)
    fig2 = sns.countplot(x= "sensor", order=sensors, data=dirty,  orient='v')
    fig2.set_xticklabels(fig2.get_xticklabels(), rotation=45)
    fig2.set(title='Totale Anzahl Bewegungen, Dataset dirty',  xlabel='Sensor', ylabel='Anzahl Bewegungen')
    fig2.axes.set_ylim(0, dirty[dirty['sensor'].isin(sensors)].sensor.value_counts()[0]+200)
    fig2.axes.set_ylabel('')

    plt.show()

def plot_sim_movement_total(sim_filtered, mal, not_o, missing=True):
    """
    Plot simultanous movement of 'Drehschrank' and 'Drehwand' for total time period
    :param sim_filtered: dataframe from filter_sim()
    :param mal: dataframe with info to malfunctions
    :param not_o: dataframe with info to occupation
    :param missing: (bool) display periods with no data caused by malfuntion or missing participants
    """
    title="Simultane Bewegungen von Drehschrank und Drehwand pro Tag"

    # data wrangling
    event_day = pd.DataFrame(sim_filtered.groupby(sim_filtered.time_of_event.dt.date)['log_sensor1'].count()).asfreq('D').reset_index()
    event_day = event_day.rename(columns={'time_of_event':'date','log_sensor1':'count'})

    # plotting
    fig = px.bar(event_day, x='date', y='count', color_discrete_sequence=['#2D92D6'])

    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title="Anzahl Bewegungen",
    )
    fig.update_traces(marker_color='#659fd6')

    if missing:
        # plot periods with missing data caused by malfunction or that the mockup hasn't been occupied
        for row in mal[mal['usable']==0].iterrows():
            fig.add_shape(
                    dict(
                        type="rect",
                        # x-reference is assigned to the x-values
                        xref="x",
                        # y-reference is assigned to the plot paper [0,1]
                        yref="paper",
                        x0=row[1][1],
                        y0=0,
                        x1=row[1][2],
                        y1=1,
                        fillcolor="LightSalmon",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    ),)
        for row in not_o.iterrows():
            fig.add_shape(
                    dict(
                        type="rect",
                        # x-reference is assigned to the x-values
                        xref="x",
                        # y-reference is assigned to the plot paper [0,1]
                        yref="paper",
                        x0=row[1][0],
                        y0=0,
                        x1=row[1][1],
                        y1=1,
                        fillcolor="LightSalmon",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    ),)

    # save plot
    save_plot(figure=fig, title=title)

    fig.show()

def plot_sim_move_hour(data, sensor1, sensor2):
    """
    Plots amount of movement per hour of day
    :param data: dataset containing columns with time_ov_event (hour) and a count.
    """
    # conditional colloration
    sns.set(rc={'figure.figsize':(20,7)})


    color='#659fd6'

    # barplot
    g = sns.barplot(x=data.time_of_event, y=data['log_sensor1'], color='#2D92D6')
    g.set(title='Anzahl simultaner Bewegungen von {} und {} pro Stunde \n Totaler Projektzeitraum'.format(sensor1, sensor2),
         xlabel = 'Uhrzeit [h]',
         ylabel = 'Anzahl Bewegungen')

    plt.show()


def plot_sim_age_situation(data_age, data_status):
    """
    Plots two barplots of datasets data_age and data_status
    """
    # set axis limit for plots
    if max(data_age['count']) >= max(data_status.occupation_period_id):
        max_val = max(data_age['count'])
    else:
        max_val = max(data_status.occupation_period_id)

    # set figure size
    sns.set(rc={'figure.figsize':(15,6)})

    # plot 1 and 2
    plt.subplot(1, 2, 1)
    fig = sns.barplot(data=data_age,x='age', y='count', color='#2D92D6' )
    fig.set(title='Simultane Bewegungen von Drehwand und Drehschrank pro Altersgruppe',
         xlabel='Altersgruppe [Jahre]',
         ylabel='Anzahl Bewegungen')
    fig.axes.set_ylim(0, max_val+50)

    plt.subplot(1, 2, 2)
    fig2 = sns.barplot(data=data_status,x='n_person', y='occupation_period_id', color='#2D92D6')
    fig2.set(title='Simultane Bewegungen von Drehwand und Drehschrank pro Benutzergruppe',
         xlabel='Anzahl Personen in Gebäude',
         ylabel='Anzahl Bewegungen')
    fig2.set_xticklabels(["Single","Couple","Mehrere [7]"])
    fig2.axes.set_ylim(0, max_val+50)


    plt.show()


# Age distribution as Pie chart
def p_age():
    df = dba.select('person')
    df.sort_values(by='age')
    values = (df['age'].value_counts())
    labels = values.index

    fig = go.Figure(data=[go.Pie(labels=labels + " Years",
                                 values=values ,
                                 textinfo='label+value+percent',
                                 textfont_size=20
                                 )])

    # save plot
    save_plot(figure=fig, title='age_pie_chart')

    fig.show()


# Sex distribution as Pie chart
def p_sex():
    df = dba.select('person')

    df = df.sort_values(by='sex')
    df = df.fillna(value="Unknown")
    values = (df['sex'].value_counts())

    fig = go.Figure(data=[go.Pie(labels=["Male", 'Female', 'Unknown'],
                                 values=values ,
                                 textinfo='label+value+percent',
                                 textfont_size=20
                                 )])
    # save plot
    save_plot(figure=fig, title='sex_pie_chart')

    fig.show()


# toDo merge p_daily, p_weekly, p_monthly, p_periodically & add nested loops for layout adding
def p_daily():

    df = time_interaction_table()
    clrs = get_config()['color']
    color = [clrs['Neutral'][0], clrs['Drehwand'][0],clrs['Drehschrank'][0],clrs['LampeDrehwand'][0],clrs['LampeAussenwand'][0]]

    data = df[['timestamp', 'total', 'Drehwand', 'Drehschrank', 'LampeDrehwand', 'LampeAussenwand']].groupby(df['timestamp'].dt.strftime("%H")).sum()
    valid_hours = df['timestamp'].loc[df['during_experiment']==True].groupby(df['timestamp'].dt.strftime("%H")).count()
    valid_hours = pd.DataFrame(valid_hours)
    data = data.merge(valid_hours, left_index=True, right_index=True).rename(columns={'total':'total_interactions','timestamp':'valid_hours'})
    data['interactions_per_hour'] = data['total_interactions']/data['valid_hours'] 
    data['Drehwand_per_hour'] = data['Drehwand'] / data['valid_hours'] 
    data['Drehschrank_per_hour'] = data['Drehschrank'] / data['valid_hours'] 
    data['LampeDrehwand_per_hour'] = data['LampeDrehwand'] / data['valid_hours'] 
    data['LampeAussenwand_per_hour'] = data['LampeAussenwand'] / data['valid_hours'] 

    fig = make_subplots(rows=5, cols=1)

    # Add Title
    title = "Aktivität nach Tageszeit"
    fig.update_layout(title_text=title, height=900)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['interactions_per_hour'],
                text=data['interactions_per_hour'].round(2),
                textposition='auto',
                name='Alle Sensoren',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[0]
            ),row=1, col=1)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Drehwand_per_hour'],
                text=data['Drehwand_per_hour'].round(2),
                textposition='auto',
                name='Drehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[1]
            ),row=2, col=1)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Drehschrank_per_hour'],
                text=data['Drehschrank_per_hour'].round(2),
                textposition='auto',
                name='Drehschrank',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[2]
            ),row=3, col=1)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['LampeDrehwand_per_hour'],
                text=data['LampeDrehwand_per_hour'].round(2),
                textposition='auto',
                name='LampeDrehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[3]
            ),row=4, col=1)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['LampeAussenwand_per_hour'],
                text=data['LampeAussenwand_per_hour'].round(2),
                textposition='auto',
                name='LampeAussenwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[4]
            ),row=5, col=1)


    fig.update_xaxes(dtick=1, row=1, col=1)
    fig.update_xaxes(dtick=1, row=2, col=1)
    fig.update_xaxes(dtick=1, row=3, col=1)
    fig.update_xaxes(dtick=1, row=4, col=1)
    fig.update_xaxes(title_text="Uhrzeit [h]", dtick=1, row=5, col=1)

    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=3, col=1)

    # save plot
    save_plot(figure=fig, title=title)

    return data, fig


def p_weekly():

    df = time_interaction_table()

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data = df[['timestamp', 'total', 'Drehwand', 'Drehschrank', 'LampeDrehwand', 'LampeAussenwand']].groupby(df['timestamp'].dt.strftime("%A")).sum().reindex(weekdays)
    valid_hours = df['timestamp'].loc[df['during_experiment']==True].groupby(df['timestamp'].dt.strftime("%A")).count().reindex(weekdays)
    valid_hours = pd.DataFrame(valid_hours)
    data = data.merge(valid_hours, left_index=True, right_index=True).rename(columns={'total':'total_interactions','timestamp':'valid_hours'})
    data['interactions_per_hour'] = data['total_interactions']/data['valid_hours'] * 24
    data['Drehwand_per_hour'] = data['Drehwand'] / data['valid_hours'] * 24
    data['Drehschrank_per_hour'] = data['Drehschrank'] / data['valid_hours'] * 24
    data['LampeDrehwand_per_hour'] = data['LampeDrehwand'] / data['valid_hours'] * 24
    data['LampeAussenwand_per_hour'] = data['LampeAussenwand'] / data['valid_hours'] * 24
    clrs = get_config()['color']
    color = [clrs['Neutral'][0], clrs['Drehwand'][0],clrs['Drehschrank'][0],clrs['LampeDrehwand'][0],clrs['LampeAussenwand'][0]]
    days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So', 'Mo ']

    fig = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], subplot_titles=("Alle Sensoren", "Drehelemente"))

    # Add Title
    title = "Aktivität nach Wochentag"
    fig.update_layout(title_text=title)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['interactions_per_hour'],
                text=data['interactions_per_hour'].round(2),
                textposition='outside',
                name='Alle Sensoren',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[0]
            ),row=1, col=1)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['Drehwand_per_hour'],
                text=data['Drehwand_per_hour'].round(2),
                textposition='outside',
                name='Drehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[1]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['Drehschrank_per_hour'],
                text=data['Drehschrank_per_hour'].round(2),
                textposition='outside',
                name='Drehschrank',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[2]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['LampeDrehwand_per_hour'],
                text=data['LampeDrehwand_per_hour'].round(2),
                textposition='outside',
                name='LampeDrehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[3]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['LampeAussenwand_per_hour'],
                text=data['LampeAussenwand_per_hour'].round(2),
                textposition='outside',
                name='LampeAussenwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[4]
            ),row=1, col=2)

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=list(
                [dict(label='Alle Drehsensoren',
                      method='update',
                      args=[{'visible': [True, True, True, True, True]},
                            # the index of True aligns with the indices of plot traces
                            {'title': 'Alle Drehsensoren',
                             'showlegend': True}], ),
                 dict(label='Drehwand',
                      method='update',
                      args=[{'visible': [True, True, False, False, False]},
                            {'title': 'Drehwand',
                             'showlegend': True}]),
                 dict(label='Drehschrank',
                      method='update',
                      args=[{'visible': [True, False, True, False, False]},
                            {'title': 'Drehschrank',
                             'showlegend': True}]),
                 dict(label='LampeDrehwand',
                      method='update',
                      args=[{'visible': [True, False, False, True, False]},
                            {'title': 'LampeDrehwand',
                             'showlegend': True}]),
                 dict(label='LampeAussenwand',
                      method='update',
                      args=[{'visible': [True, False, False, False, True]},
                            {'title': 'LampeAussenwand',
                             'showlegend': True}]),
                 ]), direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.25,
            xanchor="right",
            y=1.22,
            yanchor="top"
        )
        ])

    fig.update_xaxes(title_text="Wochentag", row=1, col=1)
    fig.update_xaxes(title_text="Wochentag", row=1, col=2)

    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=1, col=1)
    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=1, col=2)

    # save plot
    save_plot(figure=fig, title=title)

    return data, fig



def p_monthly():

    df = time_interaction_table()

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    data = df[['timestamp', 'total', 'Drehwand', 'Drehschrank', 'LampeDrehwand', 'LampeAussenwand']].groupby(df['timestamp'].dt.strftime("%B")).sum().reindex(months)
    valid_hours = df['timestamp'].loc[df['during_experiment']==True].groupby(df['timestamp'].dt.strftime("%B")).count()
    valid_hours = pd.DataFrame(valid_hours)
    data = data.merge(valid_hours, left_index=True, right_index=True).rename(columns={'total':'total_interactions','timestamp':'valid_hours'})
    data['interactions_per_hour'] = data['total_interactions']/data['valid_hours'] * 24
    data['Drehwand_per_hour'] = data['Drehwand'] / data['valid_hours'] * 24
    data['Drehschrank_per_hour'] = data['Drehschrank'] / data['valid_hours'] * 24
    data['LampeDrehwand_per_hour'] = data['LampeDrehwand'] / data['valid_hours'] * 24
    data['LampeAussenwand_per_hour'] = data['LampeAussenwand'] / data['valid_hours'] * 24

    clrs = get_config()['color']
    color = [clrs['Neutral'][0], clrs['Drehwand'][0],clrs['Drehschrank'][0],clrs['LampeDrehwand'][0],clrs['LampeAussenwand'][0]]
    data = data.rename(
        index={'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr', 'May': 'Mai', 'June': 'Jun',
               'July': 'Jul', 'August': 'Aug', 'September': 'Sep', 'October': 'Okt', 'November': 'Nov',
               'December': 'Dez'})

    fig = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], subplot_titles=("Alle Sensoren", "Drehelemente"))
    # months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez', ]

    #Add Title
    title = "Aktivität nach Monat"
    fig.update_layout(title_text=title)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['interactions_per_hour'],
                text=data['interactions_per_hour'].round(2),
                textposition='outside',
                name='Alle Sensoren',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[0]
            ),row=1, col=1)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Drehwand_per_hour'],
                text=data['Drehwand_per_hour'].round(2),
                textposition='outside',
                name='Drehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[1]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Drehschrank_per_hour'],
                text=data['Drehschrank_per_hour'].round(2),
                textposition='outside',
                name='Drehschrank',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[2]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['LampeDrehwand_per_hour'],
                text=data['LampeDrehwand_per_hour'].round(2),
                textposition='outside',
                name='LampeDrehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[3]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['LampeAussenwand_per_hour'],
                text=data['LampeAussenwand_per_hour'].round(2),
                textposition='outside',
                name='LampeAussenwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[4]
            ),row=1, col=2)

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=list(
                [dict(label='Alle Drehsensoren',
                      method='update',
                      args=[{'visible': [True, True, True, True, True]},
                            # the index of True aligns with the indices of plot traces
                            {'title': 'Alle Drehsensoren',
                             'showlegend': True}], ),
                 dict(label='Drehwand',
                      method='update',
                      args=[{'visible': [True, True, False, False, False]},
                            {'title': 'Drehwand',
                             'showlegend': True}]),
                 dict(label='Drehschrank',
                      method='update',
                      args=[{'visible': [True, False, True, False, False]},
                            {'title': 'Drehschrank',
                             'showlegend': True}]),
                 dict(label='LampeDrehwand',
                      method='update',
                      args=[{'visible': [True, False, False, True, False]},
                            {'title': 'LampeDrehwand',
                             'showlegend': True}]),
                 dict(label='LampeAussenwand',
                      method='update',
                      args=[{'visible': [True, False, False, False, True]},
                            {'title': 'LampeAussenwand',
                             'showlegend': True}]),
                 ]), direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.25,
            xanchor="right",
            y=1.22,
            yanchor="top"
        )
        ])

    fig.update_xaxes(title_text="Monat", row=1, col=1)
    fig.update_xaxes(title_text="Monat", row=1, col=2)

    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=1, col=1)
    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=1, col=2)

    # save plot
    save_plot(figure=fig, title=title)

    return data, fig

# todo move to data_wrangling.py
def t_periodically():
    df = time_interaction_table()

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    first_monday = df.loc[
        (df['timestamp'].dt.strftime("%A") != 'Monday') | (df['timestamp'].dt.strftime("%H").astype(int) >= 14)]
    second_monday = df.loc[
        (df['timestamp'].dt.strftime("%A") == 'Monday') & (df['timestamp'].dt.strftime("%H").astype(int) <= 10)]

    data = first_monday[['timestamp', 'total', 'Drehwand', 'Drehschrank', 'LampeDrehwand', 'LampeAussenwand']].groupby(
        first_monday['timestamp'].dt.strftime("%A")).sum().reindex(weekdays)
    valid_hours = first_monday['timestamp'].loc[first_monday['during_experiment'] == True].groupby(
        first_monday['timestamp'].dt.strftime("%A")).count()

    data2 = second_monday[
        ['timestamp', 'total', 'Drehwand', 'Drehschrank', 'LampeDrehwand', 'LampeAussenwand']].groupby(
        second_monday['timestamp'].dt.strftime("%A")).sum()
    valid_hours2 = second_monday['timestamp'].loc[second_monday['during_experiment'] == True].groupby(
        second_monday['timestamp'].dt.strftime("%A")).count()
    valid_hours2 = pd.DataFrame(valid_hours2)
    data2 = data2.merge(valid_hours2, left_index=True, right_index=True).rename(
        columns={'total': 'total_interactions', 'timestamp': 'valid_hours'}).reset_index()
    data2 = data2.rename(index={0: 7})

    valid_hours = pd.DataFrame(valid_hours)
    data = data.merge(valid_hours, left_index=True, right_index=True).rename(
        columns={'total': 'total_interactions', 'timestamp': 'valid_hours'}).reset_index()
    data = data.append(data2)

    data['interactions_per_hour'] = data['total_interactions'] / data['valid_hours'] * 24
    data['Drehwand_per_hour'] = data['Drehwand'] / data['valid_hours'] * 24
    data['Drehschrank_per_hour'] = data['Drehschrank'] / data['valid_hours'] * 24
    data['LampeDrehwand_per_hour'] = data['LampeDrehwand'] / data['valid_hours'] * 24
    data['LampeAussenwand_per_hour'] = data['LampeAussenwand'] / data['valid_hours'] * 24

    return data


def p_periodically():

    data = t_periodically()

    clrs = get_config()['color']
    color = [clrs['Neutral'][0], clrs['Drehwand'][0],clrs['Drehschrank'][0],clrs['LampeDrehwand'][0],clrs['LampeAussenwand'][0]]
    days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So', 'Mo ']

    fig = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], subplot_titles=("Alle Sensoren", "Drehelemente"))

    # Add Title
    title = "Aktivität nach Wochentag"
    fig.update_layout(title_text=title)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['interactions_per_hour'],
                text=data['interactions_per_hour'].round(2),
                textposition='outside',
                name='Alle Sensoren',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[0]
            ),row=1, col=1)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['Drehwand_per_hour'],
                text=data['Drehwand_per_hour'].round(2),
                textposition='outside',
                name='Drehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[1]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['Drehschrank_per_hour'],
                text=data['Drehschrank_per_hour'].round(2),
                textposition='outside',
                name='Drehschrank',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[2]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['LampeDrehwand_per_hour'],
                text=data['LampeDrehwand_per_hour'].round(2),
                textposition='outside',
                name='LampeDrehwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[3]
            ),row=1, col=2)

    fig.add_trace(
            go.Bar(
                x=days,
                y=data['LampeAussenwand_per_hour'],
                text=data['LampeAussenwand_per_hour'].round(2),
                textposition='outside',
                name='LampeAussenwand',
                showlegend=True,
                hovertemplate='%{y}',
                marker_color=color[4]
            ),row=1, col=2)

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=list(
                [dict(label='Alle Drehsensoren',
                      method='update',
                      args=[{'visible': [True, True, True, True, True]},
                            # the index of True aligns with the indices of plot traces
                            {'title': 'Alle Drehsensoren',
                             'showlegend': True}], ),
                 dict(label='Drehwand',
                      method='update',
                      args=[{'visible': [True, True, False, False, False]},
                            {'title': 'Drehwand',
                             'showlegend': True}]),
                 dict(label='Drehschrank',
                      method='update',
                      args=[{'visible': [True, False, True, False, False]},
                            {'title': 'Drehschrank',
                             'showlegend': True}]),
                 dict(label='LampeDrehwand',
                      method='update',
                      args=[{'visible': [True, False, False, True, False]},
                            {'title': 'LampeDrehwand',
                             'showlegend': True}]),
                 dict(label='LampeAussenwand',
                      method='update',
                      args=[{'visible': [True, False, False, False, True]},
                            {'title': 'LampeAussenwand',
                             'showlegend': True}]),
                 ]), direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.25,
            xanchor="right",
            y=1.22,
            yanchor="top"
        )
        ])

    fig.update_xaxes(title_text="Wochentag", row=1, col=1)
    fig.update_xaxes(title_text="Wochentag", row=1, col=2)

    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=1, col=1)
    fig.update_yaxes(title_text="Duchschnitt tägliche Interaktionen", row=1, col=2)

    # save plot
    save_plot(figure=fig, title=title + '_periodisch')

    return fig

def single_couple(df_person):
    df_pac = df_person.groupby(by = 'occupation_period_id').first().reset_index()
    df_pac = rename_person(df_pac, 'single_couple')
    plt.figure(figsize=(10, 8))
    plt.title('Verteilung Singles und Paare')
    ax = sns.countplot(x = 'single_couple', data = df_pac, palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Wohnsituation', ylabel = 'Anzahl')
    plt.show()

def single_couple_plotly(df_person):
    df_pac = df_person.groupby(by = 'occupation_period_id').first().reset_index()
    df_pac = rename_person(df_pac, 'single_couple')
    data = [
        go.Bar(
            x = df_pac['single_couple'].value_counts().keys(),
            y = df_pac['single_couple'].value_counts()
        )]
    layout = go.Layout(
        width = 600,
        title = 'Verteilung Singles und Paare',
        xaxis = dict(title = 'Wohnsituation'),
        yaxis = dict(title = 'Anzahl'),
        showlegend = False
    )
    fig = go.Figure(data = data, layout = layout)
    fig.show()

def age_dist(person):
    plt.figure(figsize=(10, 8))
    plt.title('Altersverteilung aller Teilnehmer')
    ax = sns.countplot(x = 'age', data = person, order = ['18-30', '30-45', '45-60', '60-75'], palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Alter', ylabel = 'Anzahl Personen')
    plt.show()

def sex_dist(df_person):
    df_person = df_person.copy()
    df_person = rename_person(df_person, 'sex')
    plt.figure(figsize=(10, 8))
    plt.title('Geschlechterverteilung aller Teilnehmer')
    ax = sns.countplot(x = 'Geschlecht', data = df_person, palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Geschlecht', ylabel = 'Anzahl Personen')
    plt.show()

def sex_age_dist(df_person):
    df_person = df_person.copy()
    df_person = rename_person(df_person, 'sex')
    plt.figure(figsize=(10, 8))
    plt.title('Geschlechterverteilung pro Alterskategorie')
    ax = sns.countplot(x = 'age', data = df_person, order = ['18-30', '30-45', '45-60', '60-75'], hue = 'Geschlecht', palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Alter', ylabel = 'Anzahl Personen')
    plt.legend(loc = 'upper right')
    plt.show()

def couples_sex_age(df_person):
    df_c = df_person[df_person.loc[:, 'single_couple'] == 'couple'] # c for couple
    df_c = rename_person(df_c, 'sex', 'single_couple')
    df_c['Gleichgeschlechtiges_Paar'] = df_c['Gleichgeschlechtiges_Paar'].replace(True, 'gleiches Geschlecht')
    df_c['Gleichgeschlechtiges_Paar'] = df_c['Gleichgeschlechtiges_Paar'].replace(False, 'unterschiedliches Geschlecht')
    plt.figure(figsize=(10, 8))
    plt.title('Zusammensetzung Geschlecht von Paaren pro Alterskategorie')
    ax = sns.countplot(x = 'age', data = df_c, order = ['18-30', '30-45', '45-60', '60-75'], hue = 'Gleichgeschlechtiges_Paar', palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Alter', ylabel = 'Anzahl Personen')
    plt.legend(loc = 'upper right')
    plt.show()

def singles_sex_age_dist(df_person):
    df_person_singles = df_person[df_person.single_couple == 'single']
    df_person_singles = rename_person(df_person_singles, 'sex')
    plt.figure(figsize=(10, 8))
    plt.title('Geschlechterverteilung nach Alter von Singles')
    ax = sns.countplot(x = 'age', data = df_person_singles, order = ['18-30', '30-45', '45-60', '60-75'], hue = 'Geschlecht', palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Alter', ylabel = 'Anzahl Singles')
    plt.legend(loc = 'upper right')
    plt.show()

def ss_couples_sex_age_dist(df_person):
    df_person_couples = df_person[df_person.single_sex_couple == True]
    df_person_couples = rename_person(df_person_couples, 'sex')
    plt.figure(figsize=(10, 8))
    plt.title('Geschlechterverteilung nach Alter von gleichgeschlechtrigen Paaren')
    ax = sns.countplot(x = 'age', data = df_person_couples, order = ['18-30', '30-45', '45-60', '60-75'], hue = 'Geschlecht', palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Alter', ylabel = 'Anzahl Paare')
    plt.legend(loc = 'upper right')
    plt.show()

def profession_dist(person):
    person = person.copy()
    person = rename_person(person, 'profession')
    plt.figure(figsize=(10, 8))
    plt.title('Berufsverteilung')
    ax = sns.countplot(x = 'Beruf', data = person, order = ['Student', 'Architekt', 'Andere'], palette=get_config()['color']['Neutral'])
    ax.set(xlabel = 'Beruf', ylabel = 'Anzahl Personen')
    plt.show()

def plot_turn_weights_all(sensors_all):
    """
    Plots the turn-sensor as a weighted histogram where the wheigt is the duration in 'duration_sensor_position' and the data is 'sensor_numeric2'

    :param sensors_all: dict, where key = str(sensor) and value = pd.DataFrame, log_sensor table enriched with the ocrrect 'duration_sensor_position'
    """
    fig = go.FigureWidget(layout = {'width':800, 'bargap':0})
    buttons = []

    first, vis, i = True, True, 0
    for sensor in sensors_all.keys():
        df_sens = sensors_all[sensor]
        sensor_name = df_sens.loc[:, 'sensor'].unique()[0]
        sample_size = len(df_sens.loc[:, 'occupation_period_id'].unique())
        weights = df_sens.loc[:, 'duration_sensor_position'] / sample_size
        data_sn2 = df_sens.loc[:, 'sensor_numeric2']

        data_sn2, weights, bins = add_min_max(sensor, data_sn2, weights)
        counts, bin_edges = np.histogram(data_sn2, bins = bins, weights = weights)

        clr = get_config()['color'][sensor][0]

        fig.add_trace(
            go.Bar(
                x = bin_edges,
                y = counts,
                name = sensor,
                visible = vis,
                marker_color=clr
            ))

        if first:
            first, vis = False, False
            fig.update_layout(title = 'Sensorzustand: ' + str(sample_size) + ' Singles & Paare (' + sensor_name + ')')

        visible = [False] * len(sensors_all.keys())
        visible[i] = True
        i += 1
        new_button = dict(label = sensor,
                          method = 'update',
                          args = [{'visible': visible},
                                # the index of True aligns with the indices of plot traces
                                {'title': 'Sensorzustand: ' + str(sample_size) + ' Perioden (' + sensor_name + ')',
                                 'showlegend': False}])
        buttons.append(new_button)

    # Add dropdown
    fig.update_layout(
        updatemenus = [go.layout.Updatemenu(
            active = 0,
            buttons = buttons,
            direction = 'down',
            pad = {'r': 10, 't': 10},
            showactive = True,
            x = 1,
            xanchor = 'right',
            y = 1,
            yanchor = 'bottom')],
        xaxis_title = 'Winkel (Grad)',
        yaxis_title = 'Durchschnittliche Verweildauer',
    )

    save_plot(fig, 'Drehwinkel_Alle')
    fig.show()

def plot_turn_weights_living(sensors_living):
    """
    Plots the turn-sensor as two weighted histogram where the wheigt is the duration in 'duration_sensor_position' and the data is 'sensor_numeric2'
    The first histogram shows the state of all singles, the second shows the state of all couples

    :param sensors_living: dict, where key = str(sensor) and value = pd.DataFrame log_sensor table enriched with the ocrrect 'duration_sensor_position' and 'single_couple'
    """
    sensors = sensors_living.keys()
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ('Singles', 'Paare'))
    fig.update_layout(width = 1600, bargap = 0)
    buttons = []
    first, vis = True, True

    for i, sensor in enumerate(sensors):
        df_sens = sensors_living[sensor]

        df_single = df_sens[df_sens.loc[:, 'single_couple'] == 'single']
        single_size = len(df_single.loc[:, 'occupation_period_id'].unique())
        single_weights = df_single.loc[:, 'duration_sensor_position'] / single_size
        data_single_sn2 = df_single.loc[:, 'sensor_numeric2']
        data_single_sn2, single_weights, bins = add_min_max(sensor, data_single_sn2, single_weights)
        counts_single, bin_edges_single = np.histogram(data_single_sn2, bins = bins, weights = single_weights)
        clr = get_config()['color'][sensor]


        fig.append_trace(
            go.Bar(
                x = bin_edges_single,
                y = counts_single,
                visible = vis,
                showlegend = False,
                marker_color=clr[0]
            ), 1, 1)

        df_couple = df_sens[df_sens.loc[:, 'single_couple'] == 'couple']
        couple_size = len(df_couple.loc[:, 'occupation_period_id'].unique())
        couple_weights = df_couple.loc[:, 'duration_sensor_position'] / couple_size
        data_couple_sn2 = df_couple.loc[:, 'sensor_numeric2']
        data_couple_sn2, couple_weights, bins = add_min_max(sensor, data_couple_sn2, couple_weights)
        counts_couple, bin_edges_couple = np.histogram(data_couple_sn2, bins = bins, weights = couple_weights)

        fig.append_trace(
            go.Bar(
                x = bin_edges_couple,
                y = counts_couple,
                visible = vis,
                showlegend = False,
                marker_color=clr[1]
            ), 1, 2)


        if first:
            first, vis = False, False
            fig.update_layout(title = ('Sensorzustand Wohnsituation: ' + sensor))
            fig['layout']['annotations'][0]['text'] = str(single_size) + ' Singles'
            fig['layout']['annotations'][1]['text'] = str(couple_size) + ' Paare'

        visible = [False] * len(sensors) * 2
        visible[i * 2], visible[i * 2 + 1] = True, True

        new_button = dict(label = sensor,
                          method = 'update',
                          args = [{'visible': visible},
                                  {'title': 'Sensorzustand Wohnsituation: ' + sensor,
                                   'annotations': [dict(font = dict(size = 16),
                                                        showarrow = False,
                                                        text = str(single_size) + ' Singles',
                                                        x = 0.225,
                                                        xanchor = 'center',
                                                        xref = 'paper',
                                                        y = 1.0,
                                                        yanchor = 'bottom',
                                                        yref = 'paper'),
                                                   dict(font = dict(size = 16),
                                                        showarrow = False,
                                                        text = str(couple_size) + ' Paare',
                                                        x = 0.775,
                                                        xanchor = 'center',
                                                        xref = 'paper',
                                                        y = 1.0,
                                                        yanchor = 'bottom',
                                                        yref = 'paper')]}
                                 ]
                         )

        buttons.append(new_button)

    # Add dropdown
    fig.update_layout(
        updatemenus = [go.layout.Updatemenu(
            active = 0,
            buttons = buttons,
            direction = 'down',
            pad = {'r': 10, 't': 10},
            showactive = True,
            x = 1,
            xanchor = 'right',
            y = 1.12,
            yanchor = 'bottom')]
    )
    fig['layout']['xaxis']['title'] = 'Winkel (Grad)'
    fig['layout']['xaxis2']['title'] = 'Winkel (Grad)'
    fig['layout']['yaxis']['title'] = 'Durchschnittliche Verweildauer'
    fig['layout']['yaxis2']['title'] = 'Durchschnittliche Verweildauer'

    save_plot(fig, 'Drehwinkel_Wohnsituation')
    fig.show()

def plot_turn_weights_sex(sensors_sex, couples = False):
    """
    Plots the turn-sensor as two weighted histogram where the wheigt is the duration in 'duration_sensor_position' and the data is 'sensor_numeric2'
    The first histogram shows the state of all males, the second shows the state of all females

    :param sensors_sex: dict, where key = str(sensor) and value = pd.DataFrame log_sensor table enriched with the ocrrect 'duration_sensor_position' and 'single_couple'
    :param couples: boolean, if True, it plots only couples, otherwise only singles
    """
    sensors = sensors_sex.keys()
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ('Männlich', 'Weiblich'))
    fig.update_layout(width = 1600, bargap = 0)
    buttons = []
    first, vis = True, True

    for i, sensor in enumerate(sensors):
        df_sens = sensors_sex[sensor]
        if couples:
            df_sensor = df_sens[df_sens.loc[:, 'single_couple'] == 'couple']
            cat = 'Paare'
        else:
            df_sensor = df_sens[df_sens.loc[:, 'single_couple'] == 'single']
            cat = 'Singles'

        df_m = df_sensor[df_sensor.loc[:, 'sex'] == 'm']
        m_size = len(df_m.loc[:, 'occupation_period_id'].unique())
        m_weights = df_m.loc[:, 'duration_sensor_position'] / m_size
        data_m_sn2 = df_m.loc[:, 'sensor_numeric2']
        data_m_sn2, m_weights, bins = add_min_max(sensor, data_m_sn2, m_weights)
        counts_m, bin_edges_m = np.histogram(data_m_sn2, bins = bins, weights = m_weights)
        clr = get_config()['color'][sensor]


        fig.append_trace(
            go.Bar(
                x = bin_edges_m,
                y = counts_m,
                visible = vis,
                showlegend = False,
                marker_color=clr[0]
            ), 1, 1)

        df_f = df_sensor[df_sensor.loc[:, 'sex'] == 'f']
        f_size = len(df_f.loc[:, 'occupation_period_id'].unique())
        f_weights = df_f.loc[:, 'duration_sensor_position'] / f_size
        data_f_sn2 = df_f.loc[:, 'sensor_numeric2']
        data_f_sn2, f_weights, bins = add_min_max(sensor, data_f_sn2, f_weights)
        counts_f, bin_edges_f = np.histogram(data_f_sn2, bins = bins, weights = f_weights)

        fig.append_trace(
            go.Bar(
                x = bin_edges_f,
                y = counts_f,
                visible = vis,
                showlegend = False,
                marker_color=clr[1]
            ), 1, 2)


        if first:
            first, vis = False, False
            fig.update_layout(title = ('Sensorzustand Wohnsituation ' + cat + ': ' + sensor))
            fig['layout']['annotations'][0]['text'] = str(m_size) + ' ' + cat + ' Männlich'
            fig['layout']['annotations'][1]['text'] = str(f_size) + ' ' + cat + ' Weiblich'

        visible = [False] * len(sensors) * 2
        visible[i * 2], visible[i * 2 + 1] = True, True

        new_button = dict(label = sensor,
                          method = 'update',
                          args = [{'visible': visible},
                                  {'title': 'Sensorzustand Wohnsituation ' + cat + ': ' + sensor,
                                   'annotations': [dict(font = dict(size = 16),
                                                        showarrow = False,
                                                        text = str(m_size) + ' ' + cat + ' Männlich',
                                                        x = 0.225,
                                                        xanchor = 'center',
                                                        xref = 'paper',
                                                        y = 1.0,
                                                        yanchor = 'bottom',
                                                        yref = 'paper'),
                                                   dict(font = dict(size = 16),
                                                        showarrow = False,
                                                        text = str(f_size) + ' ' + cat + ' Weiblich',
                                                        x = 0.775,
                                                        xanchor = 'center',
                                                        xref = 'paper',
                                                        y = 1.0,
                                                        yanchor = 'bottom',
                                                        yref = 'paper')]}
                                 ]
                         )

        buttons.append(new_button)

    # Add dropdown
    fig.update_layout(
        updatemenus = [go.layout.Updatemenu(
            active = 0,
            buttons = buttons,
            direction = 'down',
            pad = {'r': 10, 't': 10},
            showactive = True,
            x = 1,
            xanchor = 'right',
            y = 1.12,
            yanchor = 'bottom')]
    )
    fig['layout']['xaxis']['title'] = 'Winkel (Grad)'
    fig['layout']['xaxis2']['title'] = 'Winkel (Grad)'
    fig['layout']['yaxis']['title'] = 'Durchschnittliche Verweildauer'
    fig['layout']['yaxis2']['title'] = 'Durchschnittliche Verweildauer'

    save_plot(fig, 'Drehwinkel_Geschlecht')
    fig.show()

def plot_turn_weights_age(sensors_age, couples = False):
    """
    Plots the turn-sensor as two weighted histogram where the wheigt is the duration in 'duration_sensor_position' and the data is 'sensor_numeric2'
    The histograms show the state of all available age categories

    :param sensors_age: dict, where key = str(sensor) and value = pd.DataFrame log_sensor table enriched with the ocrrect 'duration_sensor_position' and 'single_couple'
    :param couples: boolean, if True, it plots only couples, otherwise only singles
    """
    sensors = sensors_age.keys()
    ages = ['18-30', '30-45', '45-60', '60-75']
    fig = make_subplots(rows = 2, cols = 2, subplot_titles = (ages[0], ages[1], ages[2], ages[3]), vertical_spacing = 0.15)
    fig.update_layout(height = 940)
    buttons = []
    first, vis = True, True

    annotation_template = dict(font = dict(size = 16),
                               showarrow = False,
                               text = '', # fill text
                               x = 0., # fill x
                               xanchor = 'center',
                               xref = 'paper',
                               y = 0., # fill y
                               yanchor = 'bottom',
                               yref = 'paper')

    for i, sensor in enumerate(sensors):
        annotations = []
        df_sens = sensors_age[sensor]
        clr = get_config()['color'][sensor]

        if couples:
            df_sensor = df_sens[df_sens.loc[:, 'single_couple'] == 'couple']
            cat = 'Paare'
        else:
            df_sensor = df_sens[df_sens.loc[:, 'single_couple'] == 'single']
            cat = 'Singles'

        for j, age in enumerate(ages):
            counts_age, bin_edges_age = None, None
            try: # try except if age does not occur in df_sensor
                df_age = df_sensor[df_sensor.loc[:, 'age'] == age]
                age_size = len(df_age.loc[:, 'occupation_period_id'].unique())
                age_weights = df_age.loc[:, 'duration_sensor_position'] / age_size
                data_age_sn2 = df_age.loc[:, 'sensor_numeric2']
                data_age_sn2, age_weights, bins = add_min_max(sensor, data_age_sn2, age_weights)
                counts_age, bin_edges_age = np.histogram(data_age_sn2, bins = bins, weights = age_weights)
            except:
                print('exception occured in Sensor: ' + sensor + ' ' + age)
                pass

            if j == 0:
                r_ix, c_ix = 1, 1
                x, y = fig['layout']['annotations'][0]['x'], fig['layout']['annotations'][0]['y']
            elif j == 1:
                r_ix, c_ix = 1, 2
                x, y = fig['layout']['annotations'][1]['x'], fig['layout']['annotations'][0]['y']
            elif j == 2:
                r_ix, c_ix = 2, 1
                x, y = fig['layout']['annotations'][0]['x'], fig['layout']['annotations'][2]['y']
            elif j == 3:
                r_ix, c_ix = 2, 2
                x, y = fig['layout']['annotations'][1]['x'], fig['layout']['annotations'][2]['y']

            annotation = annotation_template.copy()
            annotation['text'] = str(age_size) + ' ' + cat + ' ' + age
            annotation['x'] = x
            annotation['y'] = y
            annotations.append(annotation)

            fig.append_trace(
                go.Bar(
                    x = bin_edges_age,
                    y = counts_age,
                    visible = vis,
                    showlegend = False,
                    marker_color=clr[j]

                ), r_ix, c_ix)

            if first:
                fig['layout']['annotations'][j]['text'] = str(age_size) + ' ' + cat + ' ' + age

        if first:
            first, vis = False, False
            fig.update_layout(title = ('Sensorzustand Alter: ' + sensor))

        visible = [False] * len(sensors) * len(ages)
        for k in range(i * len(ages), (i + 1) * len(ages)):
            visible[k] = True

        new_button = dict(label = sensor,
                          method = 'update',
                          args = [{'visible': visible},
                                  {'title': 'Sensorzustand Alter: ' + sensor,
                                   'annotations': annotations}
                                 ]
                         )

        buttons.append(new_button)

    # Add dropdown
    fig.update_layout(
        updatemenus = [go.layout.Updatemenu(
            active = 0,
            buttons = buttons,
            direction = 'down',
            pad = {'r': 10, 't': 10},
            showactive = True,
            x = 1,
            xanchor = 'right',
            y = 1.05,
            yanchor = 'bottom')]
    )

    for i in range(1, len(ages) + 1): # layout axis 1 is '' and 2+ is '2', '3', etc..
        if i == 1:
            xaxis, yaxis = 'xaxis', 'yaxis'
        else:
            xaxis, yaxis = 'xaxis'+str(i), 'yaxis'+str(i)
        fig['layout'][xaxis]['title'] = 'Winkel (Grad)'
        fig['layout'][yaxis]['title'] = 'Durchschnittliche Verweildauer'

    save_plot(fig, 'Drehwinkel_Alter')
    fig.show()

def sens_weekly():
    df = dba.select('sensor_data') # get data
    person = dba.select('person')


    @interact
    def update_plotly(Sensorgruppe = ['Drehsensoren', 'Küche', 'Schubladen/Schränke', 'Türen']):
        """
        :param group: Defines sensorgroup (Dropdown to select)
        """
        group = Sensorgruppe
        clrs = list(colors.cnames.values())
         # select fitting DataFrame
        if group == 'Drehsensoren':
            data = df[df['sensor_type'] == 'turn'] # turn for 'Drehsensor'
            clrs = [clr[0] for key,clr in get_config()['color'].items()]
        elif group == 'Küche':
            data = df[df['room'] == 'K'] # K for 'Küche'

        elif group == 'Schubladen/Schränke':
            data = df[df['sensor'].isin(['S_Boden_Wand_cyr',
                                                     'S_Boden_Kueche_cyr',
                                                     'S_Schub_Wand_cyr ',
                                                     'S_Schub_Kueche_cyr',
                                                     'H_Putz_cyr',
                                                     'H_Graderobe_cyr',
                                                     'B_Schrank_cyr',
                                                     'B_Wasch_cyr',
                                                     'W_Schub_Bad_cyr',
                                                     'W_Schub_Wand_cyr',
                                                     'W_Boden_Bad_cyr',
                                                     'W_Boden_Wand_cyr'
                                                    ])]

        elif group == 'Türen':
            data = df[df['sensor'].isin(['H_Tuer_Str',
                                         'B_Tuer_Str'
                                        ])]
        # extrapolate missing data
        data = group_by_hour(data=data, sensors=data.sensor.unique())
        data['weekday'] = data.weekday.apply(dayNameFromWeekday)

        @interact
        def _update_plotly(Besetzungsperiode=(int(min(df.occupation_period_id)), int(max(df.occupation_period_id)),1), ):
            """
            Plots interaction with sensor per hour and day (select occupation period and sensor group via @interact)

            :param data: dataframe with sensor data (.select('sensor_data'))
            :param x: IntSlider to select occupation period in range min/max occupation_period
            """
            x = Besetzungsperiode
            # plotting
            kw = df[df['occupation_period_id']==x].iloc[1,:]['sensor_timestamp'].isocalendar()[1]

            data_op = data[data.occupation_period == x]
            if len(data_op) == 0: # test if data exists for this period
                print('Keine Daten für diese Kalenderwoche verfügbar')
            else:
                # user data
                data_person = person[person['occupation_period_id']==x]
                id_p = data_person.name_eth.values.tolist()
                sex = data_person.sex.values.tolist()
                age = data_person.age.values.tolist()
                id_p = str(id_p).strip('[]')
                sex = ', '.join(sex)
                age = ', '.join(age)

                display(Markdown(f"**Kalenderwoche:** {kw}, **Person/en:** {id_p}, **Altersgruppe/n:** {age}, **Geschlecht/er:** {sex}"))

                fig = px.bar(data_op, x="hour", y="count", color="sensor", barmode="group", facet_row="weekday",
                         category_orders={"weekday":["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."]},
                         color_discrete_sequence= clrs,
                         labels={ # replaces default labels by column name
                            "hour": "Uhrzeit [h]",  "count":""},
                         range_x=[0,24],
                         height=600,
                        )



                title = "Bewegungen pro Stunde und Wochentag"


                # fig.update_layout(title={'text': f"Bewegungen pro Stunde und Wochentag"})

                # remove 'column = value' notation in labels and legends
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
                # fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

                # add information below legend

                # save plot
                save_plot(figure=fig, title=title + '_periode_' + str(x))

                fig.show()

def def_sensor_age_week():
    days = ['Mo.','Di.','Mi.','Do.','Fr.','Sa.','So.']
    df_move = filter_mal(usable=1)
    sensors = sorted(df_move.sensor.unique())

    @interact(Sensor1=sensors,Sensor2=sensors)
    def _def_sensor_age_week(Sensor1='Drehwand', Sensor2='Drehschrank'):
        turn_sens = ['Drehwand', 'Drehschrank','LampeDrehwand','LampeAussenwand']
        if Sensor1 not in turn_sens:
            clr1 = get_config()['color']['Neutral'][0]
        else:
            clr1 = get_config()['color'][Sensor1]
        if Sensor2 not in turn_sens:
            clr2 = get_config()['color']['Neutral'][0]
        else:
            clr2 = get_config()['color'][Sensor2]

        df_grouped = group_norm_category(df_move[df_move.sensor.isin([Sensor1,Sensor2])], category='age', situation='single', sensor=[Sensor1,Sensor2])
        df_grouped_c = group_norm_category(df_move[df_move.sensor.isin([Sensor1,Sensor2])], category='age', situation='couple', sensor=[Sensor1,Sensor2])

        fig = px.bar(df_grouped, 
                     x='age', 
                     y='count_norm', 
                     color='sensor', 
                     barmode='group', 
                     color_discrete_map={
                        Sensor1: clr1,
                        Sensor2: clr2},
                     labels={'age':'Altersgruppe', 'count_norm':'Bewegungen'},
                     title='Bewegungen von Singles pro Altersgruppe, auf eine Woche normiert')

        save_plot(fig, f'Bewegungen von Singles pro Altersgruppe_{Sensor1}_{Sensor2}')
        fig.show()

        fig = px.bar(df_grouped_c, 
                 x='age', 
                 y='count_norm', 
                 color='sensor', 
                 barmode='group', 
                 color_discrete_map={
                        Sensor1: clr1,
                        Sensor2: clr2},
                 labels={'age':'Altersgruppe', 'count_norm':'Bewegungen'},
                 title='Bewegungen von Paaren pro Altersgruppe, auf eine Woche normiert')

        save_plot(fig, f'Bewegungen von Paaren pro Altersgruppe_{Sensor1}_{Sensor2}')
        fig.show()
        
def def_sensor_age_day():
    days = ['Mo.','Di.','Mi.','Do.','Fr.','Sa.','So.']

    df_move = filter_mal(usable=1)
    #sensors = df_move[df_move.sensor_type=='turn'].sensor.unique()
    sensors = sorted(df_move.sensor.unique())
    @interact(Sensor=sensors)
    def _def_sensor_age_day(Sensor='Drehwand'):
        clr = clrs[Sensor] if Sensor in clrs.keys() else clrs['Neutral']

        if Sensor == ['Drehwand','Drehschrank','LampeDrehwand','LampeAussenwand']:
            turn = True
        else:
            turn = False

        df_grouped = group_norm_category(df_move[df_move.sensor==Sensor], sensor=[Sensor], category='age', situation='single', day_week='day', turn=turn)
        df_grouped_c = group_norm_category(df_move[df_move.sensor==Sensor], sensor=[Sensor], category='age', situation='couple', day_week='day', turn=turn)

        fig = px.bar(df_grouped[df_grouped.sensor==Sensor],
                    x='weekday',
                    y='count_norm',
                    color = 'age',
                    color_discrete_sequence=clr,
                    barmode='group',
                    title=f'Bewegungen von Singles pro Wochentag ({Sensor}), normiert',
                    labels={'weekday':'Wochentag', 'count_norm':'Bewegungen'})
        fig.update_xaxes(
                            ticktext=["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."],
                            tickvals=[0,1,2,3,4,5,6],
                        )


        save_plot(fig, f'Bewegungen von Singles pro Wochentag ({Sensor}), normiert')
        fig.show()

        fig = px.bar(df_grouped_c[df_grouped_c.sensor==Sensor],
                    x='weekday',
                    y='count_norm',
                    color = 'age',
                    color_discrete_sequence=clr,
                    barmode='group',
                    title=f'Bewegungen von Paaren pro Wochentag ({Sensor}), normiert',
                    labels={'weekday':'Wochentag', 'count_norm':'Bewegungen'},
                    )
        fig.update_xaxes(
                            ticktext=["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."],
                            tickvals=[0,1,2,3,4,5,6],
                        )

        save_plot(fig, f'Bewegungen von Paaren pro Wochentag ({Sensor}), normiert')
        fig.show()
        
def def_sensor_sex_week():
    days = ['Mo.','Di.','Mi.','Do.','Fr.','Sa.','So.']

    df_move = filter_mal(usable=1)
    sensors = sorted(df_move.sensor.unique())

    @interact(Sensor1=sensors,Sensor2=sensors)
    def _def_sensor_sex_week(Sensor1='Drehwand', Sensor2='Drehschrank'):

        clr1 = clrs[Sensor1] if Sensor1 in clrs.keys() else clrs['Neutral']
        clr2 = clrs[Sensor2] if Sensor2 in clrs.keys() else clrs['Neutral']

        turn_sensors = ['Drehwand','Drehschrank','LampeDrehwand','LampeAussenwand']
        if Sensor1 in turn_sensors and Sensor2 in turn_sensors:
            turn = True
        else:
            turn = False

        df_grouped = group_norm_category(df_move[df_move.sensor.isin([Sensor1,Sensor2])], sensor=[Sensor1,Sensor2], category='sex', situation='single', turn=turn)
        df_grouped_c = group_norm_category(df_move[df_move.sensor.isin([Sensor1,Sensor2])], sensor=[Sensor1,Sensor2], category='sex', situation='couple', turn=turn)

        fig = px.bar(df_grouped, 
                     x='sex', 
                     y='count_norm', 
                     color='sensor',
                     color_discrete_map={
                            Sensor1: clr1,
                            Sensor2: clr2},
                     barmode='group', 
                     labels={'sex':'Geschlecht', 'count_norm':'Bewegungen'},
                     title=f'Bewegeungen von Singles pro Geschlecht ({Sensor1}, {Sensor2}), auf eine Woche normiert')
        save_plot(fig, f'Bewegeungen von Singles pro Geschlecht ({Sensor1}, {Sensor2}), normiert')


        fig.show()

        fig = px.bar(df_grouped_c, 
                     x='sex', 
                     y='count_norm', 
                     color='sensor',
                     color_discrete_map={
                            Sensor1: clr1,
                            Sensor2: clr2},  
                     barmode='group', 
                     labels={'sex':'Geschlecht', 'count_norm':'Bewegungen'},
                     title=f'Bewegeungen von Paaren pro Geschlecht ({Sensor1}, {Sensor2}), auf eine Woche normiert')
        save_plot(fig, f'Bewegeungen von Paaren pro Geschlecht ({Sensor1}, {Sensor2}), normiert')

        fig.show()
        
        
def def_sensor_sex_day():
    days = ['Mo.','Di.','Mi.','Do.','Fr.','Sa.','So.']

    df_move = filter_mal(usable=1)
    sensors = sorted(df_move.sensor.unique())
    @interact(Sensor1=sensors,Sensor2=sensors)
    def _def_sensor_sex_day(Sensor1='Drehwand'):

        clr1 = clrs[Sensor1] if Sensor1 in clrs.keys() else clrs['Neutral']

        turn_sensors = ['Drehwand','Drehschrank','LampeDrehwand','LampeAussenwand']
        if Sensor1 in turn_sensors:
            turn = True
        else:
            turn = False

        df_grouped = group_norm_category(df_move[df_move.sensor.isin([Sensor1])], sensor=[Sensor1], category='sex', situation='single', day_week='day', turn=turn)
        df_grouped_c = group_norm_category(df_move[df_move.sensor.isin([Sensor1])], sensor=[Sensor1], category='sex', situation='couple', day_week='day', turn=turn)



        fig = px.bar(df_grouped[df_grouped.sensor==Sensor1],
                    x='weekday',
                    y='count_norm',
                    color = 'sex',
                    color_discrete_sequence=clr1,
                    barmode='group',
                    title=f'Bewegungen von Singles pro Wochentag ({Sensor1}), normiert',
                    labels={'weekday':'Wochentag', 'count_norm':'Bewegungen'})
        fig.update_xaxes(
                            ticktext=["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."],
                            tickvals=[0,1,2,3,4,5,6],
                        )

        save_plot(fig, f'Bewegungen von Singles pro Wochentag ({Sensor1}), normiert')

        fig.show()

        fig = px.bar(df_grouped_c[df_grouped_c.sensor==Sensor1],
                    x='weekday',
                    y='count_norm',
                    color='sex',
                    color_discrete_sequence=clr1,
                    barmode='group',
                    title=f'Bewegungen von Paaren pro Wochentag ({Sensor1}), normiert',
                    labels={'weekday':'Wochentag', 'count_norm':'Bewegungen'})
        fig.update_xaxes(
                            ticktext=["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."],
                            tickvals=[0,1,2,3,4,5,6],
                        )
        save_plot(fig, f'Bewegungen von Paaren pro Wochentag ({Sensor1}), normiert')

        fig.show()