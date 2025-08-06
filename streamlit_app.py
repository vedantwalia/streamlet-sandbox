import streamlit as st
import requests
import plotly.graph_objs as go
from datetime import datetime
import tradermade as tm 
from dotenv import load_dotenv
import os

if not st.secrets:
    load_dotenv()
    api_key = os.getenv("TRADERMADE_API_KEY")
else:
    api_key = st.secrets["TRADERMADE_API_KEY"]

def fetch_historical_data(product, instrument, start_date, end_date):
    # API parameters and URL
    historical_url = "https://marketdata.tradermade.com/api/v1/timeseries"
    
    # Construct the query parameters
    querystring = {
        "currency": instrument,
        "api_key": api_key,
        "start_date": start_date,
        "end_date": end_date
    }
    
    # Make the API request
    response = requests.get(historical_url, params=querystring)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()['quotes']  # Extract quotes from the response
    else:
        return None