"""
Flask Application for BBRI.JK Stock Price Prediction Dashboard
"""

from flask import Flask, render_template, jsonify, request
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import HoverTool, DatetimeTickFormatter, NumeralTickFormatter, Span, Legend, LegendItem
from bokeh.resources import CDN
import pandas as pd
from datetime import datetime
import numpy as np

from data_fetcher import fetch_historical_data, prepare_data_for_model
from model_handler import predict_stock_prices

app = Flask(__name__)


def create_bokeh_chart(historical_df: pd.DataFrame, predictions: dict):
    """
    Create an interactive Bokeh chart with historical data and predictions.
    """
    # Prepare historical data
    hist_dates = pd.to_datetime(historical_df['Date'])
    hist_close = historical_df['Close'].values
    
    # Prepare prediction data
    pred_dates = pd.to_datetime(predictions['dates'])
    pred_prices = predictions['predictions']
    
    # Create figure with proper sizing
    p = figure(
        title="BBRI.JK Stock Price - Historical & Predictions",
        x_axis_type="datetime",
        width=1100,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        background_fill_color="#0d1117",
        border_fill_color="#161b22",
        outline_line_color="#30363d"
    )
    
    # Style the chart
    p.title.text_font_size = "18px"
    p.title.text_color = "#e6edf3"
    p.title.text_font = "Arial"
    
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Price (IDR)"
    p.xaxis.axis_label_text_color = "#8b949e"
    p.yaxis.axis_label_text_color = "#8b949e"
    p.xaxis.major_label_text_color = "#8b949e"
    p.yaxis.major_label_text_color = "#8b949e"
    p.xaxis.axis_line_color = "#30363d"
    p.yaxis.axis_line_color = "#30363d"
    p.xgrid.grid_line_color = "#21262d"
    p.ygrid.grid_line_color = "#21262d"
    
    # Format axes
    p.xaxis.formatter = DatetimeTickFormatter(
        days="%d %b",
        months="%b %Y"
    )
    p.yaxis.formatter = NumeralTickFormatter(format="0,0")
    
    # Plot historical data
    hist_line = p.line(
        hist_dates, hist_close,
        line_width=2,
        line_color="#58a6ff",
        legend_label="Historical Price"
    )
    
    hist_circle = p.circle(
        hist_dates, hist_close,
        size=4,
        color="#58a6ff",
        alpha=0.6
    )
    
    # Add connection line from last historical to first prediction
    connection_dates = [hist_dates.iloc[-1], pred_dates[0]]
    connection_prices = [hist_close[-1], pred_prices[0]]
    
    p.line(
        connection_dates, connection_prices,
        line_width=2,
        line_dash="dashed",
        line_color="#7ee787",
        alpha=0.8
    )
    
    # Plot predictions
    pred_line = p.line(
        pred_dates, pred_prices,
        line_width=3,
        line_color="#7ee787",
        legend_label="Predicted Price"
    )
    
    pred_circle = p.circle(
        pred_dates, pred_prices,
        size=8,
        color="#7ee787",
        fill_color="#238636",
        line_width=2
    )
    
    # Add vertical line at prediction start
    vline = Span(
        location=hist_dates.iloc[-1].timestamp() * 1000,
        dimension='height',
        line_color='#f0883e',
        line_dash='dashed',
        line_width=2,
        line_alpha=0.7
    )
    p.add_layout(vline)
    
    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Date", "@x{%F}"),
            ("Price", "Rp @y{0,0}")
        ],
        formatters={"@x": "datetime"},
        mode="vline"
    )
    p.add_tools(hover)
    
    # Style legend
    p.legend.location = "top_left"
    p.legend.background_fill_color = "#161b22"
    p.legend.background_fill_alpha = 0.8
    p.legend.border_line_color = "#30363d"
    p.legend.label_text_color = "#e6edf3"
    p.legend.title = "Legend"
    p.legend.title_text_color = "#e6edf3"
    
    return p


@app.route('/')
def index():
    """Main dashboard page."""
    try:
        # Fetch historical data
        historical_df = fetch_historical_data(days=90)
        
        # Prepare data for model
        prepared_data = prepare_data_for_model(historical_df)
        
        # Get predictions
        predictions = predict_stock_prices(prepared_data, forecast_days=7)
        
        # Create Bokeh chart
        plot = create_bokeh_chart(historical_df.tail(60), predictions)
        script, div = components(plot)
        
        # Prepare summary statistics
        last_price = float(historical_df['Close'].iloc[-1])
        pred_prices = predictions['predictions']
        
        summary_stats = {
            'last_price': last_price,
            'pred_min': min(pred_prices),
            'pred_max': max(pred_prices),
            'pred_avg': sum(pred_prices) / len(pred_prices),
            'change': ((pred_prices[-1] - last_price) / last_price) * 100,
            'trend': 'up' if pred_prices[-1] > last_price else 'down'
        }
        
        # Format prediction table data
        prediction_table = []
        for i, (date, price) in enumerate(zip(predictions['dates'], predictions['predictions'])):
            change = ((price - last_price) / last_price) * 100
            daily_change = 0 if i == 0 else ((price - predictions['predictions'][i-1]) / predictions['predictions'][i-1]) * 100
            prediction_table.append({
                'date': date,
                'day': datetime.strptime(date, '%Y-%m-%d').strftime('%A'),
                'price': price,
                'change': change,
                'daily_change': daily_change
            })
        
        return render_template(
            'index.html',
            bokeh_script=script,
            bokeh_div=div,
            bokeh_resources=CDN.render(),
            predictions=prediction_table,
            summary_stats=summary_stats,
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/api/predict')
def api_predict():
    """API endpoint for predictions."""
    try:
        days = request.args.get('days', 7, type=int)
        days = min(max(days, 1), 30)  # Limit to 1-30 days
        
        historical_df = fetch_historical_data(days=90)
        prepared_data = prepare_data_for_model(historical_df)
        predictions = predict_stock_prices(prepared_data, forecast_days=days)
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': [
                    {'date': d, 'price': p}
                    for d, p in zip(predictions['dates'], predictions['predictions'])
                ],
                'last_actual': {
                    'date': predictions['last_actual_date'],
                    'price': predictions['last_actual_price']
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/historical')
def api_historical():
    """API endpoint for historical data."""
    try:
        days = request.args.get('days', 60, type=int)
        days = min(max(days, 7), 365)  # Limit to 7-365 days
        
        historical_df = fetch_historical_data(days=days)
        
        return jsonify({
            'success': True,
            'data': {
                'historical': [
                    {
                        'date': row['Date'].strftime('%Y-%m-%d'),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume'])
                    }
                    for _, row in historical_df.iterrows()
                ]
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting BBRI.JK Stock Prediction Dashboard...")
    print("Access the dashboard at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
