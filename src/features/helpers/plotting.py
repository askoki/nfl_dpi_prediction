import plotly.express as px
import pandas as pd


def animated_slice(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='team',
        text='position',
        animation_frame='time',
        animation_group='displayName',
        range_x=[-10, 130],
        range_y=[-10, 60],
        hover_data=['displayName', 'jerseyNumber', 's', 'a', 'dis', 'o', 'dir', 'playDirection'])
    fig.update_traces(textposition='top center', marker_size=10)
    fig.update_layout(paper_bgcolor='darkgreen', plot_bgcolor='darkgreen', font_color='white')
    return fig
