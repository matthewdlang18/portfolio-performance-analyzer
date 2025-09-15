import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import os
import re
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import warnings

# Page configuration
st.set_page_config(
    page_title="Portfolio Performance Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        border: 1px solid #d6d9dc;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .positive { color: #00C851; }
    .negative { color: #ff4444; }
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
</style>
""",
    unsafe_allow_html=True,
)

def load_portfolio_data():
    """Load and preprocess portfolio data from CSV file"""
    try:
        # Try multiple path strategies to find the CSV file
        possible_paths = [
            'Investment_income_balance_detail.csv',  # Current working directory
            os.path.join(os.path.dirname(__file__), 'Investment_income_balance_detail.csv'),  # Script directory
            os.path.join(os.getcwd(), 'Investment_income_balance_detail.csv'),  # Current working directory
            '/Users/mattlang/VSCode/metrics/Investment_income_balance_detail.csv'  # Absolute path
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            st.error("❌ Could not find Investment_income_balance_detail.csv file. Please ensure it exists in the project directory.")
            return None
            
        df = pd.read_csv(csv_path)
        
        # Clean the data
        def clean_currency(value):
            if pd.isna(value):
                return 0
            # Remove quotes, dollar signs, commas, and parentheses
            value = str(value).replace('"', '').replace('$', '').replace(',', '')
            # Handle negative values in parentheses
            if '(' in value and ')' in value:
                value = '-' + value.replace('(', '').replace(')', '')
            try:
                return float(value)
            except:
                return 0
        
        # Parse date
        def parse_date(date_str):
            if 'Aug 2025' in str(date_str):
                return '2025-12-31'
            elif '-' in str(date_str):
                # Handle format like "Jul-25"
                parts = str(date_str).split('-')
                if len(parts) == 2:
                    month_map = {
                        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                    }
                    month = month_map.get(parts[0], '01')
                    year = f"20{parts[1]}" if len(parts[1]) == 2 else parts[1]
                    # Use last day of month for monthly data
                    day = '31' if month in ['01', '03', '05', '07', '08', '10', '12'] else '30'
                    if month == '02':
                        day = '28'
                    return f"{year}-{month}-{day}"
            return None
        
        # Process the data
        df['Date'] = df['Monthly'].apply(parse_date)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Beginning_Balance'] = df['Beginning balance'].apply(clean_currency)
        df['Ending_Balance'] = df['Ending balance'].apply(clean_currency)
        df['Deposits'] = df['Deposits'].apply(clean_currency)
        df['Withdrawals'] = df['Withdrawals'].apply(clean_currency)
        df['Market_Change'] = df['Market change'].apply(clean_currency)
        df['Dividends'] = df['Dividends'].apply(clean_currency)
        
        # Remove invalid data and sort
        df = df.dropna(subset=['Date']).sort_values('Date', ascending=False)
        
        # Calculate additional fields for compatibility
        df = df.rename(columns={
            'Date': 'date',
            'Beginning_Balance': 'beginBalance', 
            'Ending_Balance': 'endBalance',
            'Deposits': 'deposits',
            'Withdrawals': 'withdrawals'
        })
        
        return df[['date', 'beginBalance', 'endBalance', 'deposits', 'withdrawals']].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading portfolio data: {str(e)}")
        return None

def load_portfolio_positions():
    """Load and preprocess portfolio positions from CSV file"""
    try:
        # Try multiple path strategies to find the CSV file
        possible_paths = [
            'Portfolio_Positions_Aug-11-2025.csv',  # Current working directory
            os.path.join(os.path.dirname(__file__), 'Portfolio_Positions_Aug-11-2025.csv'),  # Script directory
            os.path.join(os.getcwd(), 'Portfolio_Positions_Aug-11-2025.csv'),  # Current working directory
            '/Users/mattlang/VSCode/metrics/Portfolio_Positions_Aug-11-2025.csv'  # Absolute path
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            st.error("❌ Could not find Portfolio_Positions_Aug-11-2025.csv file. Please ensure it exists in the project directory.")
            return None
        
        df = pd.read_csv(csv_path)
        
        # Filter out cash positions and clean the data
        df = df[df['Symbol'].notna() & (df['Symbol'] != 'SPAXX**')].copy()
        
        # Clean currency values
        def clean_currency(value):
            if pd.isna(value) or value == '':
                return 0
            # Remove quotes, dollar signs, commas, plus signs
            value = str(value).replace('"', '').replace('$', '').replace(',', '').replace('+', '')
            try:
                return float(value)
            except:
                return 0
        
        # Clean percentage values
        def clean_percentage(value):
            if pd.isna(value) or value == '':
                return 0
            # Remove quotes, percent signs, plus signs
            value = str(value).replace('"', '').replace('%', '').replace('+', '')
            try:
                return float(value) / 100
            except:
                return 0
        
        # Clean the relevant columns
        df['Current_Value'] = df['Current Value'].apply(clean_currency)
        df['Last_Price'] = df['Last Price'].apply(clean_currency)
        df['Percent_Of_Account'] = df['Percent Of Account'].apply(clean_percentage)
        df['Total_Gain_Loss_Dollar'] = df['Total Gain/Loss Dollar'].apply(clean_currency)
        df['Total_Gain_Loss_Percent'] = df['Total Gain/Loss Percent'].apply(clean_percentage)
        
        # Ensure numeric columns are properly typed (including Quantity)
        numeric_columns = ['Quantity', 'Last_Price', 'Current_Value', 'Percent_Of_Account', 'Total_Gain_Loss_Dollar', 'Total_Gain_Loss_Percent']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Select and rename columns for easier use
        portfolio_df = df[['Symbol', 'Description', 'Quantity', 'Last_Price', 'Current_Value', 
                          'Percent_Of_Account', 'Total_Gain_Loss_Dollar', 'Total_Gain_Loss_Percent']].copy()
        
        # Ensure all numeric columns are properly typed for Arrow serialization
        for col in numeric_columns:
            if col in portfolio_df.columns:
                portfolio_df[col] = portfolio_df[col].astype('float64')
        
        # Ensure string columns are properly typed
        portfolio_df['Symbol'] = portfolio_df['Symbol'].astype('string')
        portfolio_df['Description'] = portfolio_df['Description'].astype('string')
        
        return portfolio_df
        
    except Exception as e:
        st.error(f"Error loading portfolio positions: {str(e)}")
        return None

def get_spx_data_dynamic(start_date, end_date):
    """Fetch SPX data dynamically from Yahoo Finance for any date range"""
    try:
        import yfinance as yf
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spx_ticker = yf.Ticker("^GSPC")  # S&P 500 index
            hist_data = spx_ticker.history(start=start_date, end=end_date, interval="1d")
        
        if hist_data.empty:
            return None
        
        # Resample to month-end data and create the required format
        monthly_data = hist_data['Close'].resample('ME').last()
        
        # Convert to timezone-naive to avoid comparison issues
        monthly_index = monthly_data.index.tz_convert(None) if monthly_data.index.tz is not None else monthly_data.index
        
        df = pd.DataFrame({
            'date': monthly_index,
            'price': monthly_data.values
        })
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching SPX data: {str(e)}")
        return None

def get_spxe_data_dynamic(start_date, end_date):
    """Fetch SPXE data dynamically from Yahoo Finance for any date range"""
    try:
        import yfinance as yf
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spxe_ticker = yf.Ticker("RSP")  # Invesco S&P 500 Equal Weight ETF (alternative to SPXE)
            hist_data = spxe_ticker.history(start=start_date, end=end_date, interval="1d")
        
        if hist_data.empty:
            return None
        
        # Resample to month-end data and create the required format
        monthly_data = hist_data['Close'].resample('ME').last()
        
        # Convert to timezone-naive to avoid comparison issues
        monthly_index = monthly_data.index.tz_convert(None) if monthly_data.index.tz is not None else monthly_data.index
        
        df = pd.DataFrame({
            'date': monthly_index,
            'price': monthly_data.values
        })
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching SPXE data: {str(e)}")
        return None

def get_spx_monthly_data():
    """Return SPX monthly data - fallback static data for offline use"""
    data = [
        {'date': '2021-01-29', 'price': 3714.24},
        {'date': '2021-02-26', 'price': 3811.15},
        {'date': '2021-03-31', 'price': 3972.89},
        {'date': '2021-04-30', 'price': 4181.17},
        {'date': '2021-05-28', 'price': 4204.11},
        {'date': '2021-06-30', 'price': 4297.5},
        {'date': '2021-07-30', 'price': 4395.26},
        {'date': '2021-08-31', 'price': 4522.68},
        {'date': '2021-09-30', 'price': 4307.54},
        {'date': '2021-10-29', 'price': 4605.38},
        {'date': '2021-11-30', 'price': 4567.0},
        {'date': '2021-12-31', 'price': 4766.18},
        {'date': '2022-01-31', 'price': 4515.55},
        {'date': '2022-02-28', 'price': 4373.94},
        {'date': '2022-03-31', 'price': 4530.41},
        {'date': '2022-04-29', 'price': 4131.93},
        {'date': '2022-05-31', 'price': 4132.15},
        {'date': '2022-06-30', 'price': 3785.38},
        {'date': '2022-07-29', 'price': 4130.29},
        {'date': '2022-08-31', 'price': 3955.0},
        {'date': '2022-09-30', 'price': 3585.62},
        {'date': '2022-10-31', 'price': 3871.98},
        {'date': '2022-11-30', 'price': 4080.11},
        {'date': '2022-12-30', 'price': 3839.5},
        {'date': '2023-01-31', 'price': 4076.6},
        {'date': '2023-02-28', 'price': 3970.04},
        {'date': '2023-03-31', 'price': 4109.31},
        {'date': '2023-04-28', 'price': 4169.48},
        {'date': '2023-05-31', 'price': 4179.83},
        {'date': '2023-06-30', 'price': 4450.38},
        {'date': '2023-07-31', 'price': 4588.96},
        {'date': '2023-08-31', 'price': 4507.66},
        {'date': '2023-09-29', 'price': 4288.05},
        {'date': '2023-10-31', 'price': 4193.8},
        {'date': '2023-11-30', 'price': 4567.8},
        {'date': '2023-12-29', 'price': 4769.83},
        {'date': '2024-01-31', 'price': 4845.65},
        {'date': '2024-02-29', 'price': 5096.27},
        {'date': '2024-03-28', 'price': 5254.35},
        {'date': '2024-04-30', 'price': 5035.69},
        {'date': '2024-05-31', 'price': 5277.51},
        {'date': '2024-06-28', 'price': 5460.48},
        {'date': '2024-07-31', 'price': 5522.3},
        {'date': '2024-08-30', 'price': 5648.4},
        {'date': '2024-09-30', 'price': 5762.48},
        {'date': '2024-10-31', 'price': 5705.45},
        {'date': '2024-11-29', 'price': 6032.38},
        {'date': '2024-12-31', 'price': 5881.22},
        {'date': '2025-01-31', 'price': 6090.27},
        {'date': '2025-02-28', 'price': 5970.84},
        {'date': '2025-03-31', 'price': 5254.35},
        {'date': '2025-04-30', 'price': 5469.30},
        {'date': '2025-05-30', 'price': 5932.85},
        {'date': '2025-06-30', 'price': 6279.36},
        {'date': '2025-07-31', 'price': 6238.0},
        {'date': '2025-08-30', 'price': 6449.8},
        {'date': '2025-09-30', 'price': 6598.0},
        {'date': '2025-10-31', 'price': 6723.0},
        {'date': '2025-11-29', 'price': 6850.0},
        {'date': '2025-12-31', 'price': 6978.0}
    ]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_spxe_monthly_data():
    """Return SPXE monthly data - fallback static data for offline use"""
    data = [
        {'date': '2021-01-29', 'price': 40.17},
        {'date': '2021-02-26', 'price': 41.17},
        {'date': '2021-03-31', 'price': 42.78},
        {'date': '2021-04-30', 'price': 44.92},
        {'date': '2021-05-28', 'price': 45.21},
        {'date': '2021-06-30', 'price': 46.19},
        {'date': '2021-07-30', 'price': 47.12},
        {'date': '2021-08-31', 'price': 48.42},
        {'date': '2021-09-30', 'price': 46.05},
        {'date': '2021-10-29', 'price': 49.24},
        {'date': '2021-11-30', 'price': 48.84},
        {'date': '2021-12-31', 'price': 50.95},
        {'date': '2022-01-31', 'price': 48.27},
        {'date': '2022-02-28', 'price': 46.71},
        {'date': '2022-03-31', 'price': 48.38},
        {'date': '2022-04-29', 'price': 44.15},
        {'date': '2022-05-31', 'price': 44.20},
        {'date': '2022-06-30', 'price': 40.45},
        {'date': '2022-07-29', 'price': 44.13},
        {'date': '2022-08-31', 'price': 42.25},
        {'date': '2022-09-30', 'price': 38.26},
        {'date': '2022-10-31', 'price': 41.36},
        {'date': '2022-11-30', 'price': 43.58},
        {'date': '2022-12-30', 'price': 41.00},
        {'date': '2023-01-31', 'price': 43.54},
        {'date': '2023-02-28', 'price': 42.39},
        {'date': '2023-03-31', 'price': 43.88},
        {'date': '2023-04-28', 'price': 44.54},
        {'date': '2023-05-31', 'price': 44.64},
        {'date': '2023-06-30', 'price': 47.52},
        {'date': '2023-07-31', 'price': 49.00},
        {'date': '2023-08-31', 'price': 48.17},
        {'date': '2023-09-29', 'price': 45.78},
        {'date': '2023-10-31', 'price': 44.77},
        {'date': '2023-11-30', 'price': 48.75},
        {'date': '2023-12-29', 'price': 50.95},
        {'date': '2024-01-31', 'price': 51.76},
        {'date': '2024-02-29', 'price': 54.44},
        {'date': '2024-03-28', 'price': 56.16},
        {'date': '2024-04-30', 'price': 53.82},
        {'date': '2024-05-31', 'price': 56.33},
        {'date': '2024-06-28', 'price': 58.29},
        {'date': '2024-07-31', 'price': 58.98},
        {'date': '2024-08-30', 'price': 60.31},
        {'date': '2024-09-30', 'price': 61.53},
        {'date': '2024-10-31', 'price': 60.93},
        {'date': '2024-11-29', 'price': 64.43},
        {'date': '2024-12-31', 'price': 62.83},
        {'date': '2025-01-31', 'price': 65.05},
        {'date': '2025-02-28', 'price': 63.75},
        {'date': '2025-03-31', 'price': 56.16},
        {'date': '2025-04-30', 'price': 58.40},
        {'date': '2025-05-30', 'price': 63.34},
        {'date': '2025-06-30', 'price': 67.09},
        {'date': '2025-07-31', 'price': 66.64},
        {'date': '2025-08-30', 'price': 69.27},
        {'date': '2025-09-30', 'price': 70.85},
        {'date': '2025-10-31', 'price': 72.15},
        {'date': '2025-11-29', 'price': 73.50},
        {'date': '2025-12-31', 'price': 74.92}
    ]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_index_returns(price_data, start_date, end_date):
    """Calculate monthly returns from price data for index funds"""
    # Normalize dates to handle timezone issues
    start_dt = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(start_date).tz is None else pd.to_datetime(start_date).tz_convert(None)
    end_dt = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(end_date).tz is None else pd.to_datetime(end_date).tz_convert(None)
    
    # Ensure price_data dates are timezone-naive for comparison
    price_data_copy = price_data.copy()
    if price_data_copy['date'].dt.tz is not None:
        price_data_copy['date'] = price_data_copy['date'].dt.tz_convert(None)
    
    filtered_data = price_data_copy[
        (price_data_copy['date'] >= start_dt) & 
        (price_data_copy['date'] <= end_dt)
    ].copy()
    
    if len(filtered_data) < 2:
        return None
    
    # Calculate monthly returns
    returns = []
    for i in range(1, len(filtered_data)):
        return_value = (filtered_data.iloc[i]['price'] - filtered_data.iloc[i-1]['price']) / filtered_data.iloc[i-1]['price']
        returns.append({
            'date': filtered_data.iloc[i]['date'],
            'return': return_value,
            'returnPct': return_value * 100
        })
    
    return_values = [r['return'] for r in returns]
    mean_return = pd.Series(return_values).mean()
    monthly_std_dev = pd.Series(return_values).std(ddof=1)
    
    # Annualized metrics
    annualized_return = mean_return * 12
    annualized_std_dev = monthly_std_dev * (12 ** 0.5)
    
    # Total return
    total_return = (filtered_data.iloc[-1]['price'] - filtered_data.iloc[0]['price']) / filtered_data.iloc[0]['price']
    
    return {
        'monthlyReturns': returns,
        'meanReturn': mean_return,
        'monthlyStdDev': monthly_std_dev,
        'annualizedReturn': annualized_return,
        'annualizedStdDev': annualized_std_dev,
        'totalReturn': total_return,
        'totalMonths': len(return_values),
        'startPrice': filtered_data.iloc[0]['price'],
        'endPrice': filtered_data.iloc[-1]['price']
    }

def calculate_portfolio_returns(portfolio_data, start_date, end_date):
    """Calculate time-weighted returns for portfolio"""
    filtered_data = portfolio_data[
        (portfolio_data['date'] >= pd.to_datetime(start_date)) & 
        (portfolio_data['date'] <= pd.to_datetime(end_date))
    ].copy().sort_values('date')
    
    if len(filtered_data) < 2:
        return None
    
    # Calculate monthly returns using time-weighted method
    returns = []
    for i in range(1, len(filtered_data)):
        current = filtered_data.iloc[i]
        previous = filtered_data.iloc[i-1]
        
        net_deposits = current['deposits'] - current['withdrawals']
        time_weighted_return = (current['endBalance'] - previous['endBalance'] - net_deposits) / previous['endBalance']
        
        returns.append({
            'date': current['date'],
            'return': time_weighted_return,
            'returnPct': time_weighted_return * 100
        })
    
    return_values = [r['return'] for r in returns]
    mean_return = pd.Series(return_values).mean()
    monthly_std_dev = pd.Series(return_values).std(ddof=1)
    
    # Annualized metrics
    annualized_return = mean_return * 12
    annualized_std_dev = monthly_std_dev * (12 ** 0.5)
    
    # Total return using compound returns
    total_return = 1
    for monthly_return in return_values:
        total_return *= (1 + monthly_return)
        total_return *= (1 + monthly_return)
    total_return -= 1
    
    return {
        'monthlyReturns': returns,
        'meanReturn': mean_return,
        'monthlyStdDev': monthly_std_dev,
        'annualizedReturn': annualized_return,
        'annualizedStdDev': annualized_std_dev,
        'totalReturn': total_return,
        'totalMonths': len(return_values)
    }

def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate):
    """Calculate Sharpe ratio"""
    if annualized_volatility == 0:
        return 0
    return (annualized_return - risk_free_rate) / annualized_volatility

def create_cml_plot(portfolio_return, portfolio_volatility, spx_return, spx_volatility, risk_free_rate):
    """Create Capital Market Line plot that goes through SPX and risk-free rate"""
    # Calculate the market Sharpe ratio using SPX
    market_sharpe_ratio = calculate_sharpe_ratio(spx_return, spx_volatility, risk_free_rate)
    
    # Create a range of volatilities for the CML
    max_vol = max(portfolio_volatility * 1.5, spx_volatility * 1.5, 0.3)
    vol_range = np.linspace(0, max_vol, 100)
    
    # Calculate expected returns along the CML using market Sharpe ratio
    cml_returns = risk_free_rate + market_sharpe_ratio * vol_range
    
    # Create the plot
    fig = go.Figure()
    
    # Add the Capital Market Line
    fig.add_trace(go.Scatter(
        x=vol_range * 100,  # Convert to percentage
        y=cml_returns * 100,  # Convert to percentage
        mode='lines',
        name='Capital Market Line',
        line=dict(color='darkblue', width=3)
    ))
    
    # Add the risk-free asset point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[risk_free_rate * 100],
        mode='markers',
        name='Risk-Free Asset',
        marker=dict(color='green', size=12, symbol='circle')
    ))
    
    # Add the market portfolio (SPX) point
    fig.add_trace(go.Scatter(
        x=[spx_volatility * 100],
        y=[spx_return * 100],
        mode='markers',
        name='Market Portfolio (SPX)',
        marker=dict(color='red', size=14, symbol='star')
    ))
    
    # Add the Portfolio point
    fig.add_trace(go.Scatter(
        x=[portfolio_volatility * 100],
        y=[portfolio_return * 100],
        mode='markers',
        name='Your Portfolio',
        marker=dict(color='blue', size=12, symbol='diamond')
    ))
    
    fig.update_layout(
        title='Capital Market Line - Optimal Risk-Return Trade-off',
        xaxis_title='Annualized Volatility (%)',
        yaxis_title='Expected Return (%)',
        height=500,
        showlegend=True,
        hovermode='closest',
        annotations=[
            dict(
                x=spx_volatility * 100,
                y=spx_return * 100 + 2,
                text=f"Market Sharpe: {market_sharpe_ratio:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="white",
                bordercolor="red"
            )
        ]
    )
    
    return fig

def get_current_fcf(symbol):
    """Get the most current FCF using TTM from quarterly data instead of outdated annual data"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to get TTM FCF from quarterly financials first (most current)
        try:
            quarterly_cf = ticker.quarterly_cashflow
            if quarterly_cf is not None and not quarterly_cf.empty:
                # Get the last 4 quarters for TTM calculation
                fcf_rows = quarterly_cf.loc[quarterly_cf.index == 'Free Cash Flow']
                if not fcf_rows.empty and len(fcf_rows.columns) >= 4:
                    ttm_fcf = fcf_rows.iloc[0, :4].sum()
                    if not pd.isna(ttm_fcf) and ttm_fcf != 0:
                        return ttm_fcf, "ttm_quarterly"
        except:
            pass
        
        # Fallback to calculated TTM from operating cash flow - capex
        try:
            quarterly_cf = ticker.quarterly_cashflow
            if quarterly_cf is not None and not quarterly_cf.empty and len(quarterly_cf.columns) >= 4:
                # Get operating cash flow
                op_cf_rows = quarterly_cf.loc[quarterly_cf.index == 'Operating Cash Flow']
                capex_rows = quarterly_cf.loc[quarterly_cf.index == 'Capital Expenditure']
                
                if not op_cf_rows.empty and not capex_rows.empty:
                    ttm_op_cf = op_cf_rows.iloc[0, :4].sum()
                    ttm_capex = capex_rows.iloc[0, :4].sum()
                    
                    if not pd.isna(ttm_op_cf) and not pd.isna(ttm_capex):
                        calculated_fcf = ttm_op_cf + ttm_capex  # CapEx is negative
                        if calculated_fcf != 0:
                            return calculated_fcf, "ttm_calculated"
        except:
            pass
        
        # Final fallback to annual data from info
        info = ticker.info
        annual_fcf = info.get('freeCashflow')
        if annual_fcf and annual_fcf != 0:
            return annual_fcf, "annual_info"
        
        return None, "no_data"
        
    except Exception as e:
        return None, f"error: {str(e)}"

def fetch_stock_data(symbol):
    """Fetch comprehensive stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current FCF using improved method
        current_fcf, fcf_method = get_current_fcf(symbol)
        
        # Extract key financial metrics
        data = {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', symbol)),
            'price': info.get('currentPrice', info.get('regularMarketPrice')),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'ev_revenue': info.get('enterpriseToRevenue'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'free_cash_flow': current_fcf,
            'fcf_method': fcf_method,
            'operating_cash_flow': info.get('operatingCashflow'),
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            'beta': info.get('beta'),
            'week_52_high': info.get('fiftyTwoWeekHigh'),
            'week_52_low': info.get('fiftyTwoWeekLow'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'revenue': info.get('totalRevenue'),
            'gross_profit': info.get('grossProfits'),
            'ebitda': info.get('ebitda')
        }
        
        # Calculate P/FCF ratio manually
        if data['market_cap'] and data['free_cash_flow'] and data['free_cash_flow'] > 0:
            data['p_fcf_ratio'] = data['market_cap'] / data['free_cash_flow']
        else:
            data['p_fcf_ratio'] = None
            
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_dcf_valuation(symbol, growth_rate=0.08, growth_years=5, terminal_growth=0.025, wacc=0.10):
    """Calculate DCF valuation for a stock"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get financial data
        current_fcf, fcf_method = get_current_fcf(symbol)
        shares_outstanding = info.get('sharesOutstanding')
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        
        if not current_fcf or not shares_outstanding:
            return None, "Missing required financial data"
        
        # Calculate projected cash flows
        projected_fcf = []
        base_fcf = current_fcf
        
        for year in range(1, growth_years + 1):
            base_fcf *= (1 + growth_rate)
            projected_fcf.append(base_fcf)
        
        # Terminal value
        terminal_fcf = base_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        
        # Discount to present value
        total_pv = 0
        for year, fcf in enumerate(projected_fcf, 1):
            pv = fcf / ((1 + wacc) ** year)
            total_pv += pv
        
        # Discount terminal value
        pv_terminal = terminal_value / ((1 + wacc) ** growth_years)
        total_pv += pv_terminal
        
        # Calculate per-share value
        enterprise_value = total_pv
        net_debt = total_debt - total_cash
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        # Valuation analysis
        valuation_result = {
            'symbol': symbol,
            'current_price': current_price,
            'intrinsic_value': intrinsic_value_per_share,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'upside_downside': (intrinsic_value_per_share - current_price) / current_price if current_price else None,
            'assumptions': {
                'growth_rate': growth_rate,
                'growth_years': growth_years,
                'terminal_growth': terminal_growth,
                'wacc': wacc,
                'current_fcf': current_fcf,
                'fcf_method': fcf_method
            }
        }
        
        return valuation_result, "success"
        
    except Exception as e:
        return None, f"Error in DCF calculation: {str(e)}"

def get_portfolio_beta_analysis(portfolio_positions):
    """Calculate portfolio beta vs SPX benchmark"""
    try:
        if portfolio_positions is None or len(portfolio_positions) == 0:
            return None
        
        # Get SPX data for beta calculation
        spx = yf.Ticker("^GSPC")
        spx_hist = spx.history(period="2y")
        spx_returns = spx_hist['Close'].pct_change().dropna()
        
        portfolio_beta_data = []
        total_value = portfolio_positions['Current_Value'].sum()
        
        for _, position in portfolio_positions.iterrows():
            try:
                ticker = yf.Ticker(position['Symbol'])
                hist = ticker.history(period="2y")
                
                if len(hist) > 50:  # Ensure sufficient data
                    stock_returns = hist['Close'].pct_change().dropna()
                    
                    # Align dates and calculate beta
                    common_dates = stock_returns.index.intersection(spx_returns.index)
                    if len(common_dates) > 50:
                        aligned_stock = stock_returns.loc[common_dates]
                        aligned_spx = spx_returns.loc[common_dates]
                        
                        # Calculate beta
                        covariance = np.cov(aligned_stock, aligned_spx)[0][1]
                        market_variance = np.var(aligned_spx)
                        beta = covariance / market_variance if market_variance != 0 else 1.0
                        
                        # Calculate correlation
                        correlation = np.corrcoef(aligned_stock, aligned_spx)[0][1]
                        
                        weight = position['Current_Value'] / total_value
                        
                        portfolio_beta_data.append({
                            'Symbol': position['Symbol'],
                            'Weight': weight,
                            'Beta': beta,
                            'Correlation': correlation,
                            'Current_Value': position['Current_Value'],
                            'Weighted_Beta': weight * beta
                        })
                        
            except Exception as e:
                st.warning(f"Could not calculate beta for {position['Symbol']}: {str(e)}")
                continue
        
        if portfolio_beta_data:
            beta_df = pd.DataFrame(portfolio_beta_data)
            portfolio_beta = beta_df['Weighted_Beta'].sum()
            
            return {
                'portfolio_beta': portfolio_beta,
                'individual_betas': beta_df,
                'total_positions': len(beta_df),
                'coverage': len(beta_df) / len(portfolio_positions)
            }
        
        return None
        
    except Exception as e:
        st.error(f"Error calculating portfolio beta: {str(e)}")
        return None

def format_currency(value, decimal_places=2):
    """Format currency values with appropriate scaling"""
    if pd.isna(value) or value == 0:
        return "N/A"
    
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value/1e12:.{decimal_places}f}T"
    elif abs_value >= 1e9:
        return f"${value/1e9:.{decimal_places}f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.{decimal_places}f}M"
    elif abs_value >= 1e3:
        return f"${value/1e3:.{decimal_places}f}K"
    else:
        return f"${value:.{decimal_places}f}"

def format_percentage(value, decimal_places=2):
    """Format percentage values (expects decimal format like 0.0379 for 3.79%)"""
    if pd.isna(value):
        return "N/A"
    return f"{value*100:.{decimal_places}f}%"

def format_dividend_yield(value, decimal_places=2):
    """Format dividend yield values (Yahoo Finance returns them as percentages already)"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.{decimal_places}f}%"

def get_historical_data_custom_range(tickers, start_date, end_date):
    """Fetch historical price data for specific date range"""
    try:
        # Filter out any None or empty tickers
        valid_tickers = [t for t in tickers if t and isinstance(t, str) and t.strip()]
        
        if not valid_tickers:
            return None, "No valid tickers provided"
        
        # Download data for the specified date range
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(
                valid_tickers, 
                start=start_date, 
                end=end_date, 
                interval="1d", 
                group_by='ticker', 
                auto_adjust=True, 
                progress=False
            )
        
        if data.empty:
            return None, "No data downloaded from Yahoo Finance for the specified date range"
        
        # Handle different data structures based on number of tickers
        price_data = pd.DataFrame()
        
        if len(valid_tickers) == 1:
            ticker = valid_tickers[0]
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-index case
                if 'Close' in data.columns.levels[1]:
                    close_col = [col for col in data.columns if col[1] == 'Close'][0]
                    price_data[ticker] = data[close_col]
            else:
                # Single index case
                if 'Close' in data.columns:
                    price_data[ticker] = data['Close']
                else:
                    return None, f"No close price data found for {ticker}"
        else:
            # Multiple tickers
            failed_tickers = []
            
            for ticker in valid_tickers:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        # Multi-index case: (ticker, 'Close')
                        if (ticker, 'Close') in data.columns:
                            price_data[ticker] = data[(ticker, 'Close')]
                        else:
                            failed_tickers.append(ticker)
                    else:
                        # Single index case (shouldn't happen with multiple tickers, but handle it)
                        if 'Close' in data.columns:
                            price_data[ticker] = data['Close']
                        else:
                            failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
            
            if failed_tickers:
                st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")
        
        # Clean the data
        # Remove any columns with all NaN values
        price_data = price_data.dropna(axis=1, how='all')
        
        # Forward fill missing values, then backward fill
        price_data = price_data.ffill().bfill()
        
        # Drop any remaining rows with NaN values
        price_data = price_data.dropna()
        
        if price_data.empty:
            return None, "No valid price data after cleaning"
        
        # Calculate the actual months of data
        months_of_data = len(price_data.resample('ME').last())
        
        # Ensure we have enough data points (at least 50 trading days and 12 months)
        if len(price_data) < 50:
            return None, f"Insufficient data: only {len(price_data)} trading days available"
        
        if months_of_data < 12:
            return None, f"Insufficient data: only {months_of_data} months available, minimum 12 months required"
        
        st.success(f"✅ Successfully fetched {len(price_data)} days of data ({months_of_data} months) for {len(price_data.columns)} stocks")
        
        return price_data, "success"
        
    except Exception as e:
        return None, f"Error fetching historical data: {str(e)}"

def get_historical_data(tickers, years=2):
    """Fetch historical price data for multiple tickers"""
    try:
        # Filter out any None or empty tickers
        valid_tickers = [t for t in tickers if t and isinstance(t, str) and t.strip()]
        
        if not valid_tickers:
            return None, "No valid tickers provided"
        
        # Convert years to period string
        period_str = f"{years}y"
        
        # Download data for all tickers at once
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(valid_tickers, period=period_str, interval="1d", group_by='ticker', auto_adjust=True, progress=False)
        
        if data.empty:
            return None, "No data downloaded from Yahoo Finance"
        
        # Handle different data structures based on number of tickers
        price_data = pd.DataFrame()
        
        if len(valid_tickers) == 1:
            ticker = valid_tickers[0]
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-index case
                if 'Close' in data.columns.levels[1]:
                    close_col = [col for col in data.columns if col[1] == 'Close'][0]
                    price_data[ticker] = data[close_col]
            else:
                # Single index case
                if 'Close' in data.columns:
                    price_data[ticker] = data['Close']
                else:
                    return None, f"No close price data found for {ticker}"
        else:
            # Multiple tickers
            failed_tickers = []
            
            for ticker in valid_tickers:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        # Multi-index case: (ticker, 'Close')
                        if (ticker, 'Close') in data.columns:
                            price_data[ticker] = data[(ticker, 'Close')]
                        else:
                            failed_tickers.append(ticker)
                    else:
                        # Single index case (shouldn't happen with multiple tickers, but handle it)
                        if 'Close' in data.columns:
                            price_data[ticker] = data['Close']
                        else:
                            failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
            
            if failed_tickers:
                st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")
        
        # Clean the data
        # Remove any columns with all NaN values
        price_data = price_data.dropna(axis=1, how='all')
        
        # Forward fill missing values, then backward fill
        price_data = price_data.ffill().bfill()
        
        # Drop any remaining rows with NaN values
        price_data = price_data.dropna()
        
        if price_data.empty:
            return None, "No valid price data after cleaning"
        
        # Ensure we have enough data points (at least 50 trading days)
        if len(price_data) < 50:
            return None, f"Insufficient data: only {len(price_data)} trading days available"
        
        st.success(f"✅ Successfully fetched {len(price_data)} days of data for {len(price_data.columns)} stocks")
        
        return price_data, "success"
        
    except Exception as e:
        return None, f"Error fetching historical data: {str(e)}"

def calculate_returns_and_stats(price_data):
    """Calculate returns, mean returns, and covariance matrix following efficient.md methodology"""
    try:
        # Calculate monthly returns from daily prices
        # Resample to month-end prices, then calculate returns
        monthly_prices = price_data.resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        
        # Ensure we have enough monthly data points (at least 24 months)
        if len(monthly_returns) < 24:
            return None, None, None, "Insufficient data: need at least 24 months for reliable optimization"
        
        # Annualize returns using arithmetic mean approach to match manual calculations
        # Your formula: (1 + average_monthly_return)^12 - 1
        # Calculate arithmetic mean for each stock, then annualize
        monthly_means = monthly_returns.mean()
        annual_returns = ((1 + monthly_means) ** 12) - 1
        
        # Annualize covariance matrix (multiply by 12 for monthly data)
        annual_cov_matrix = monthly_returns.cov() * 12
        
        # Validate covariance matrix is positive definite
        eigenvalues = np.linalg.eigvals(annual_cov_matrix)
        if np.any(eigenvalues <= 0):
            return None, None, None, "Covariance matrix is not positive definite"
        
        return monthly_returns, annual_returns, annual_cov_matrix, "success"
        
    except Exception as e:
        return None, None, None, f"Error calculating statistics: {str(e)}"

def optimize_portfolio(annual_returns, cov_matrix, risk_free_rate=0.03, num_portfolios=50):
    """Generate efficient frontier portfolios following efficient.md methodology"""
    try:
        num_assets = len(annual_returns)
        
        # Validation checks as per efficient.md
        if num_assets < 2:
            return None, "Need at least 2 assets for optimization"
        
        # Check for extreme volatility (monthly volatility > 200% would be extreme)
        monthly_vol_threshold = 2.0
        annual_vols = np.sqrt(np.diag(cov_matrix))
        if np.any(annual_vols > monthly_vol_threshold):
            st.warning("⚠️ Extremely volatile data detected in some assets")
        
        # Step 1: Generate target returns from min to max individual asset return
        min_return = annual_returns.min()
        max_return = annual_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        # Step 2: Optimize for each target return using efficient.md methodology
        def minimize_variance_for_target(target_return):
            """Minimize portfolio variance for a given target return"""
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints as per efficient.md
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.dot(x, annual_returns) - target_return}  # target return
            ]
            
            # Bounds (long-only constraint)
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/num_assets] * num_assets)
            
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP',
                bounds=bounds, 
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                portfolio_volatility = np.sqrt(result.fun)
                sharpe_ratio = (target_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return {
                    'return': target_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': result.x
                }
            else:
                return None
        
        # Step 3: Build efficient frontier
        efficient_portfolios = []
        for target_return in target_returns:
            portfolio = minimize_variance_for_target(target_return)
            if portfolio is not None:
                efficient_portfolios.append(portfolio)
        
        if not efficient_portfolios:
            return None, "Failed to generate any efficient portfolios"
        
        # Find minimum variance portfolio (first valid portfolio should be close)
        min_vol_portfolio = min(efficient_portfolios, key=lambda p: p['volatility'])
        
        # Find maximum Sharpe ratio portfolio using direct optimization (Method 1 from efficient.md)
        def maximize_sharpe_objective(weights):
            """Objective: maximize Sharpe ratio = minimize negative Sharpe ratio"""
            portfolio_return = np.dot(weights, annual_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 1e10  # Avoid division by zero
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe_ratio  # Minimize negative Sharpe
        
        constraints_sharpe = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds_sharpe = tuple((0, 1) for _ in range(num_assets))
        initial_weights_sharpe = np.array([1/num_assets] * num_assets)
        
        max_sharpe_result = minimize(
            maximize_sharpe_objective, 
            initial_weights_sharpe, 
            method='SLSQP',
            bounds=bounds_sharpe, 
            constraints=constraints_sharpe,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not max_sharpe_result.success:
            return None, "Failed to find maximum Sharpe ratio portfolio"
        
        # Calculate max Sharpe portfolio metrics
        max_sharpe_weights = max_sharpe_result.x
        max_sharpe_return = np.dot(max_sharpe_weights, annual_returns)
        max_sharpe_volatility = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
        max_sharpe_sharpe = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility
        
        max_sharpe_portfolio = {
            'return': max_sharpe_return,
            'volatility': max_sharpe_volatility,
            'weights': max_sharpe_weights,
            'sharpe_ratio': max_sharpe_sharpe
        }
        
        # Validation checks as per efficient.md
        def validate_optimization_results(portfolio, portfolio_name):
            weights = portfolio['weights']
            
            # Check weights sum to 1
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1) > 1e-6:
                st.warning(f"⚠️ {portfolio_name}: Weights don't sum to 1 ({weight_sum:.6f})")
            
            # Check no negative weights (long-only)
            if np.any(weights < -1e-6):
                st.warning(f"⚠️ {portfolio_name}: Negative weights detected")
            
            # Sanity checks
            vol = portfolio['volatility']
            ret = portfolio['return']
            if not (0 < vol < 1):
                st.warning(f"⚠️ {portfolio_name}: Unrealistic volatility: {vol:.3f}")
            if not (-0.5 < ret < 2):
                st.warning(f"⚠️ {portfolio_name}: Unrealistic return: {ret:.3f}")
        
        # Validate results
        validate_optimization_results(min_vol_portfolio, "Min Volatility Portfolio")
        validate_optimization_results(max_sharpe_portfolio, "Max Sharpe Portfolio")
        
        # Prepare results
        results = {
            'efficient_frontier': efficient_portfolios,
            'min_volatility_portfolio': min_vol_portfolio,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'statistics': {
                'num_portfolios_generated': len(efficient_portfolios),
                'target_return_range': (min_return, max_return),
                'individual_asset_returns': annual_returns.tolist(),
                'individual_asset_volatilities': np.sqrt(np.diag(cov_matrix)).tolist()
            }
        }
        
        return results, "success"
        
    except Exception as e:
        return None, f"Error in portfolio optimization: {str(e)}"

# CAMP-Based Efficient Frontier Functions
def calculate_caamp_risk_components(returns, market_returns, tickers_list):
    """Calculate CAPM risk components: betas, R-squared, and idiosyncratic variances"""
    
    # Calculate market variance (annualized)
    market_var = market_returns.var() * 252
    market_mean = market_returns.mean() * 252
    
    caamp_data = {
        'betas': {},
        'r_squared': {},
        'idiosyncratic_vars': {},
        'market_var': market_var,
        'market_mean': market_mean
    }
    
    for ticker in tickers_list:
        if ticker in returns.columns:
            stock_returns = returns[ticker].dropna()
            aligned_market = market_returns.reindex(stock_returns.index).dropna()
            aligned_stock = stock_returns.reindex(aligned_market.index).dropna()
            
            if len(aligned_stock) > 20:  # Need sufficient data points
                # Calculate beta
                covariance = np.cov(aligned_stock, aligned_market)[0, 1] * 252
                beta = covariance / market_var
                
                # Calculate R-squared (correlation coefficient squared)
                correlation = np.corrcoef(aligned_stock, aligned_market)[0, 1]
                r_squared = correlation ** 2
                
                # Calculate idiosyncratic variance
                # Idiosyncratic variance = (1 - R²) × Total variance
                total_var = aligned_stock.var() * 252
                idiosyncratic_var = (1 - r_squared) * total_var
                
                caamp_data['betas'][ticker] = beta
                caamp_data['r_squared'][ticker] = r_squared
                caamp_data['idiosyncratic_vars'][ticker] = max(idiosyncratic_var, 0.001)  # Minimum variance floor
            else:
                # Fallback values for insufficient data
                caamp_data['betas'][ticker] = 1.0
                caamp_data['r_squared'][ticker] = 0.5
                caamp_data['idiosyncratic_vars'][ticker] = 0.04  # 20% annual volatility
    
    return caamp_data

def caamp_portfolio_variance(weights, caamp_data, tickers_list):
    """Calculate portfolio variance using CAMP risk model"""
    
    # Calculate portfolio beta
    portfolio_beta = sum(weights[i] * caamp_data['betas'].get(tickers_list[i], 1.0) 
                        for i in range(len(weights)))
    
    # Systematic variance (market risk)
    systematic_var = (portfolio_beta ** 2) * caamp_data['market_var']
    
    # Idiosyncratic variance (diversifiable risk)
    idiosyncratic_var = sum(weights[i] ** 2 * caamp_data['idiosyncratic_vars'].get(tickers_list[i], 0.04)
                           for i in range(len(weights)))
    
    total_variance = systematic_var + idiosyncratic_var
    return total_variance

def optimize_caamp_portfolio(expected_returns, caamp_data, tickers_list, target_return=None, risk_free_rate=0.045):
    """Optimize portfolio using CAAMP risk model"""
    
    n_assets = len(expected_returns)
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Add target return constraint if specified
    if target_return is not None:
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: np.dot(x, expected_returns) - target_return
        })
    
    # Bounds: 0 <= weight <= 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    x0 = np.array([1/n_assets] * n_assets)
    
    if target_return is None:
        # Maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = caamp_portfolio_variance(weights, caamp_data, tickers_list)
            portfolio_std = np.sqrt(portfolio_var)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio  # Minimize negative Sharpe ratio
    else:
        # Minimize variance for target return
        def objective(weights):
            return caamp_portfolio_variance(weights, caamp_data, tickers_list)
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_var = caamp_portfolio_variance(optimal_weights, caamp_data, tickers_list)
        portfolio_std = np.sqrt(portfolio_var)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'success': True
        }
    else:
        return {'success': False, 'message': result.message}

def generate_caamp_efficient_frontier(expected_returns, caamp_data, tickers_list, risk_free_rate=0.045, num_portfolios=50):
    """Generate CAAMP-based efficient frontier"""
    
    # Calculate range of target returns
    min_return = min(expected_returns)
    max_return = max(expected_returns)
    target_returns = np.linspace(min_return, max_return, num_portfolios)
    
    efficient_portfolios = []
    
    for target_return in target_returns:
        result = optimize_caamp_portfolio(expected_returns, caamp_data, tickers_list, target_return, risk_free_rate)
        
        if result['success']:
            efficient_portfolios.append({
                'return': result['return'],
                'volatility': result['volatility'],
                'sharpe_ratio': result['sharpe_ratio'],
                'weights': result['weights']
            })
    
    # Find optimal Sharpe ratio portfolio
    optimal_result = optimize_caamp_portfolio(expected_returns, caamp_data, tickers_list, None, risk_free_rate)
    
    return {
        'efficient_portfolios': efficient_portfolios,
        'optimal_portfolio': optimal_result if optimal_result['success'] else None
    }

# Beta Strategy Comparison Functions
def calculate_beta_strategy_weights(annual_returns, cov_matrix, beta_data, tickers_list, risk_free_rate=0.03, market_variance=0.04):
    """Calculate portfolio weights that maximize Sharpe ratio using TRUE beta-based portfolio variance (not covariance matrix)"""
    from scipy.optimize import minimize
    
    try:
        # Validate inputs
        if not beta_data:
            return [1.0/len(tickers_list)] * len(tickers_list)
        
        if len(tickers_list) == 0:
            return []
        
        # Get beta values for available tickers
        available_tickers = [t for t in tickers_list if t in beta_data]
        n_stocks = len(tickers_list)
        
        if len(available_tickers) == 0:
            return [1.0/n_stocks] * n_stocks

        # Ensure inputs are numpy arrays
        annual_returns = np.array(annual_returns)
        cov_matrix = np.array(cov_matrix)
        
        # Validate array dimensions
        if len(annual_returns) != n_stocks:
            return [1.0/n_stocks] * n_stocks
            
        if cov_matrix.shape != (n_stocks, n_stocks):
            return [1.0/n_stocks] * n_stocks

        # Estimate market variance from S&P 500 historical data if not provided
        if market_variance == 0.04:  # Default value, try to get real market variance
            try:
                # Use SPX data to calculate market variance
                import datetime
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=365*2)  # 2 years of data
                
                # Try to get actual market variance
                try:
                    import yfinance as yf
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        market_data = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", progress=False)
                        if not market_data.empty:
                            market_monthly = market_data['Close'].resample('ME').last()
                            market_returns = market_monthly.pct_change().dropna()
                            if len(market_returns) > 0:  # Check length instead of using Series in condition
                                market_variance = float(market_returns.var() * 12)  # Annualize and convert to float
                                if market_variance > 0:
                                    market_variance = float(market_variance)
                            else:
                                pass  # Use default
                except Exception as inner_e:
                    pass  # Use default
            except Exception as outer_e:
                pass  # Use default of 0.04 (20% annual volatility)

        def calculate_beta_portfolio_variance(weights):
            """Calculate portfolio variance using PURE beta methodology"""
            # Portfolio beta (weighted average of individual betas)
            portfolio_beta = 0.0
            for i, ticker in enumerate(tickers_list):
                if ticker in beta_data:
                    portfolio_beta += weights[i] * beta_data[ticker]['beta']
                else:
                    portfolio_beta += weights[i] * 1.0  # Default beta
            
            # Systematic variance component: (Portfolio Beta)² × Market Variance
            systematic_variance = (portfolio_beta ** 2) * market_variance
            
            # Idiosyncratic variance component: sum of (weight² × idiosyncratic_variance)
            idiosyncratic_variance = 0.0
            for i, ticker in enumerate(tickers_list):
                if ticker in beta_data:
                    # Calculate idiosyncratic variance from R-squared
                    r_squared = beta_data[ticker].get('r_squared', 0.5)
                    # Get stock's total variance from annual returns (if possible)
                    stock_return = annual_returns[i]
                    # Estimate total variance from covariance matrix diagonal as backup
                    total_var = cov_matrix[i, i]
                    # Idiosyncratic variance = (1 - R²) × Total Variance
                    idio_var = (1 - r_squared) * total_var
                    idiosyncratic_variance += (weights[i] ** 2) * idio_var
                else:
                    # Default: assume 50% systematic, 50% idiosyncratic
                    estimated_total_var = cov_matrix[i, i]
                    idio_var = 0.5 * estimated_total_var
                    idiosyncratic_variance += (weights[i] ** 2) * idio_var
            
            # Total portfolio variance = systematic + idiosyncratic
            total_variance = systematic_variance + idiosyncratic_variance
            return max(total_variance, 1e-8)  # Ensure positive

        def maximize_sharpe_with_beta_objective(weights):
            """Objective: maximize Sharpe ratio using beta-based portfolio variance"""
            portfolio_return = np.dot(weights, annual_returns)
            
            # Use BETA-BASED variance calculation (NOT covariance matrix)
            portfolio_variance = calculate_beta_portfolio_variance(weights)
            portfolio_vol = np.sqrt(portfolio_variance)
            
            if portfolio_vol == 0:
                return 1e10  # Avoid division by zero
            
            # Calculate Sharpe ratio using beta-derived volatility
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
            
            return -sharpe_ratio  # Minimize negative Sharpe (maximize Sharpe)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_stocks))  # long-only
        initial_weights = np.array([1/n_stocks] * n_stocks)

        result = minimize(
            maximize_sharpe_with_beta_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            weights = result.x.tolist()
            return weights
        else:
            # Fallback to equal weights
            return [1.0/n_stocks] * n_stocks
            
    except Exception as e:
        # Fallback to equal weights on error
        try:
            return [1.0/len(tickers_list)] * len(tickers_list)
        except:
            return []

def calculate_portfolio_beta(weights, beta_data, tickers_list):
    """Calculate portfolio beta as weighted average of individual stock betas"""
    portfolio_beta = 0.0
    for i, ticker in enumerate(tickers_list):
        if ticker in beta_data:
            portfolio_beta += weights[i] * beta_data[ticker]['beta']
        else:
            portfolio_beta += weights[i] * 1.0  # Default beta of 1.0
    return portfolio_beta

def display_optimal_beta_strategy(optimization_results, beta_data, annual_returns, cov_matrix, risk_free_rate, tickers_list):
    """Display both Markowitz optimal and the beta-optimized Sharpe strategy"""
    try:
        st.markdown("### 🎯 **Beta Strategy Analysis**")
        st.markdown("Compare Markowitz optimization with beta-aware Sharpe ratio optimization:")
        
        # Validate inputs
        if not beta_data:
            st.error("❌ No beta data available for analysis")
            return False
        
        if len(tickers_list) == 0:
            st.error("❌ No tickers provided for analysis")
            return False
        
        if annual_returns is None or len(annual_returns) == 0:
            st.error("❌ No annual returns data available")
            return False
        
        if cov_matrix is None or cov_matrix.size == 0:
            st.error("❌ No covariance matrix data available")
            return False
        
        # Get Markowitz results for comparison
        if isinstance(optimization_results['max_sharpe_portfolio']['weights'], dict):
            markowitz_weights = list(optimization_results['max_sharpe_portfolio']['weights'].values())
        else:
            markowitz_weights = optimization_results['max_sharpe_portfolio']['weights']
        
        # Validate markowitz weights
        if len(markowitz_weights) != len(tickers_list):
            st.error(f"❌ Markowitz weights length ({len(markowitz_weights)}) doesn't match tickers length ({len(tickers_list)})")
            return False
        
        # Calculate beta-optimized strategy (maximizes Sharpe with beta awareness)
        with st.spinner("Calculating beta-optimized portfolio weights..."):
            beta_optimal_weights = calculate_beta_strategy_weights(
                annual_returns, cov_matrix, beta_data, tickers_list, risk_free_rate
            )
        
        # Validate beta optimal weights
        if beta_optimal_weights is None or len(beta_optimal_weights) != len(tickers_list):
            st.error(f"❌ Beta optimization failed to produce valid weights")
            return False
            
        # Ensure weights are numpy arrays for calculations
        markowitz_weights = np.array(markowitz_weights)
        beta_optimal_weights = np.array(beta_optimal_weights)
        annual_returns = np.array(annual_returns)
        
        st.success("✅ Beta optimization completed successfully!")
        
        # Calculate metrics for both strategies
        markowitz_beta = calculate_portfolio_beta(markowitz_weights, beta_data, tickers_list)
        markowitz_return = np.dot(markowitz_weights, annual_returns)
        markowitz_vol = np.sqrt(np.dot(markowitz_weights, np.dot(cov_matrix, markowitz_weights)))
        markowitz_sharpe = (markowitz_return - risk_free_rate) / markowitz_vol if markowitz_vol > 0 else 0
        
        beta_optimal_beta = calculate_portfolio_beta(beta_optimal_weights, beta_data, tickers_list)
        beta_optimal_return = np.dot(beta_optimal_weights, annual_returns)
        
        # Calculate beta-optimized portfolio volatility using TRUE beta-based method
        # Portfolio beta
        portfolio_beta = beta_optimal_beta
        
        # Estimate market variance (same method as in calculate_beta_strategy_weights)
        market_variance = 0.04  # Default 20% annual volatility
        try:
            import datetime
            import yfinance as yf
            import warnings
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=365*2)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                market_data = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", progress=False)
                if not market_data.empty:
                    market_monthly = market_data['Close'].resample('ME').last()
                    market_returns = market_monthly.pct_change().dropna()
                    if len(market_returns) > 0:  # Check length instead of using Series in condition
                        market_variance = float(market_returns.var() * 12)  # Annualize and convert to float
                        if market_variance > 0:
                            market_variance = float(market_variance)
        except Exception as e:
            pass  # Use default
        
        # Systematic variance: (Portfolio Beta)² × Market Variance
        systematic_variance = (portfolio_beta ** 2) * market_variance
        
        # Convert covariance matrix to numpy array for easier indexing
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix_np = cov_matrix.values
        else:
            cov_matrix_np = np.array(cov_matrix)
        
        # Idiosyncratic variance: sum of (weight² × idiosyncratic_variance)
        idiosyncratic_variance = 0.0
        for i, ticker in enumerate(tickers_list):
            weight = beta_optimal_weights[i]
            if ticker in beta_data:
                r_squared = beta_data[ticker].get('r_squared', 0.5)
                total_var = cov_matrix_np[i, i]  # Use numpy array indexing
                idio_var = (1 - r_squared) * total_var
                idiosyncratic_variance += (weight ** 2) * idio_var
            else:
                # Default: assume 50% idiosyncratic
                estimated_total_var = cov_matrix_np[i, i]  # Use numpy array indexing
                idio_var = 0.5 * estimated_total_var
                idiosyncratic_variance += (weight ** 2) * idio_var
        
        # Total beta-based portfolio variance
        beta_portfolio_variance = systematic_variance + idiosyncratic_variance
        beta_optimal_vol = np.sqrt(max(beta_portfolio_variance, 1e-8))
        beta_optimal_sharpe = (beta_optimal_return - risk_free_rate) / beta_optimal_vol if beta_optimal_vol > 0 else 0
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 **Markowitz Optimal**")
            st.metric("Expected Return", f"{markowitz_return*100:.2f}%")
            st.metric("Volatility", f"{markowitz_vol*100:.2f}%")
            st.metric("Sharpe Ratio", f"{markowitz_sharpe:.3f}")
            st.metric("Portfolio Beta", f"{markowitz_beta:.3f}")
            
            # Show top Markowitz holdings
            markowitz_weights_df = pd.DataFrame({
                'Stock': tickers_list,
                'Weight': markowitz_weights
            }).sort_values('Weight', ascending=False)
            
            significant_markowitz = markowitz_weights_df[markowitz_weights_df['Weight'] > 0.005]
            
            st.markdown("**Top Holdings (>0.5%)**")
            for _, row in significant_markowitz.head(5).iterrows():
                st.write(f"• {row['Stock']}: {row['Weight']*100:.1f}%")
        
        with col2:
            st.markdown("#### 🎯 **Beta-Optimized Sharpe**")
            st.metric("Expected Return", f"{beta_optimal_return*100:.2f}%")
            st.metric("Volatility", f"{beta_optimal_vol*100:.2f}%")
            st.metric("Sharpe Ratio", f"{beta_optimal_sharpe:.3f}")
            st.metric("Portfolio Beta", f"{beta_optimal_beta:.3f}")
            
            # Show top beta strategy holdings
            beta_weights_df = pd.DataFrame({
                'Stock': tickers_list,
                'Weight': beta_optimal_weights
            }).sort_values('Weight', ascending=False)
            
            significant_beta = beta_weights_df[beta_weights_df['Weight'] > 0.005]
            
            st.markdown("**Top Holdings (>0.5%)**")
            for _, row in significant_beta.head(5).iterrows():
                st.write(f"• {row['Stock']}: {row['Weight']*100:.1f}%")
        
        # Performance comparison
        st.markdown("#### 📊 **Performance Comparison**")
        
        # Determine winner
        if markowitz_sharpe > beta_optimal_sharpe:
            st.success(f"🏆 **Winner: Markowitz Optimal** (Sharpe: {markowitz_sharpe:.3f} vs {beta_optimal_sharpe:.3f})")
            winner_explanation = "The traditional Markowitz optimization provides better risk-adjusted returns."
        else:
            st.success(f"🏆 **Winner: Beta-Optimized Sharpe** (Sharpe: {beta_optimal_sharpe:.3f} vs {markowitz_sharpe:.3f})")
            winner_explanation = "The beta-aware optimization strategy outperforms traditional Markowitz optimization."
        
        st.info(winner_explanation)
        
        # Create side-by-side weight comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=tickers_list,
            y=[w*100 for w in markowitz_weights],
            name='Markowitz Optimal',
            marker_color='#2563eb',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            x=tickers_list,
            y=[w*100 for w in beta_optimal_weights],
            name='Beta-Optimized Sharpe',
            marker_color='#dc2626',
            opacity=0.8
        ))
        
        fig.update_layout(
            title='Portfolio Weight Comparison: Markowitz (Covariance Matrix) vs Beta-Based Variance Optimization',
            xaxis_title='Stocks',
            yaxis_title='Weight (%)',
            barmode='group',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy explanation
        with st.expander("📚 Strategy Explanations"):
            st.markdown("**Markowitz Optimal**: Traditional mean-variance optimization that maximizes Sharpe ratio using full covariance matrix.")
            st.markdown("**Beta-Optimized Sharpe**: Enhanced Sharpe ratio optimization using **true beta-based portfolio variance calculation**:")
            st.markdown("  - Portfolio Variance = (Portfolio Beta)² × Market Variance + Σ(weight² × idiosyncratic_variance)")
            st.markdown("  - Systematic Risk: Captured through portfolio beta vs S&P 500")
            st.markdown("  - Idiosyncratic Risk: Individual stock-specific risk (1 - R²) × stock variance")
            st.markdown("  - **No covariance matrix used** - pure beta methodology")
            
            if beta_optimal_beta < 0.8:
                st.info("🛡️ **Beta Strategy Risk Profile**: Defensive portfolio providing downside protection during market declines.")
            elif beta_optimal_beta > 1.2:
                st.warning("⚡ **Beta Strategy Risk Profile**: Aggressive portfolio amplifying market movements for higher risk/reward.")
            else:
                st.info("⚖️ **Beta Strategy Risk Profile**: Balanced portfolio with market-like risk exposure.")
        
        # Store the beta strategy in session state for Monte Carlo
        st.session_state.best_beta_strategy = {
            'name': 'Beta-Optimized Sharpe',
            'weights': beta_optimal_weights,
            'metrics': {
                'return': beta_optimal_return,
                'volatility': beta_optimal_vol,
                'sharpe': beta_optimal_sharpe,
                'beta': beta_optimal_beta
            }
        }
        
        return True
        
    except Exception as e:
        st.error(f"❌ **Error in beta analysis**: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        
        # Add more detailed debugging information
        try:
            st.error(f"Beta data keys: {list(beta_data.keys()) if beta_data else 'No beta data'}")
            st.error(f"Tickers list: {tickers_list}")
            st.error(f"Annual returns shape: {np.array(annual_returns).shape if annual_returns is not None else 'None'}")
            st.error(f"Covariance matrix shape: {np.array(cov_matrix).shape if cov_matrix is not None else 'None'}")
        except:
            st.error("Additional debugging information could not be displayed")
        
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return False

def create_efficient_frontier_plot(results, annual_returns, risk_free_rate, asset_names, caamp_results=None):
    """Create efficient frontier visualization with optional CAAMP frontier following efficient.md structure"""
    try:
        efficient_portfolios = results['efficient_frontier']
        cov_matrix = results.get('cov_matrix')
        
        if not efficient_portfolios:
            return None
        
        # Extract data for plotting Markowitz frontier
        returns = [p['return'] * 100 for p in efficient_portfolios]  # Convert to percentage
        volatilities = [p['volatility'] * 100 for p in efficient_portfolios]  # Convert to percentage
        sharpe_ratios = [p['sharpe_ratio'] for p in efficient_portfolios]
        
        # Create the plot
        fig = go.Figure()
        
        # Add Markowitz efficient frontier
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='lines+markers',
            name='Markowitz Efficient Frontier',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=6),
            hovertemplate='Markowitz<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{customdata:.3f}<extra></extra>',
            customdata=sharpe_ratios
        ))
        
        # Add CAAMP efficient frontier if available
        if caamp_results and caamp_results['efficient_portfolios']:
            caamp_returns = [p['return'] * 100 for p in caamp_results['efficient_portfolios']]
            caamp_volatilities = [p['volatility'] * 100 for p in caamp_results['efficient_portfolios']]
            caamp_sharpe_ratios = [p['sharpe_ratio'] for p in caamp_results['efficient_portfolios']]
            
            fig.add_trace(go.Scatter(
                x=caamp_volatilities,
                y=caamp_returns,
                mode='lines+markers',
                name='CAAMP Efficient Frontier',
                line=dict(color='#dc2626', width=3, dash='dash'),
                marker=dict(size=6, symbol='square'),
                hovertemplate='CAAMP<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{customdata:.3f}<extra></extra>',
                customdata=caamp_sharpe_ratios
            ))
        
        # Add individual assets using statistics from results
        individual_returns = [ret * 100 for ret in results['statistics']['individual_asset_returns']]
        individual_volatilities = [vol * 100 for vol in results['statistics']['individual_asset_volatilities']]
        
        fig.add_trace(go.Scatter(
            x=individual_volatilities,
            y=individual_returns,
            mode='markers',
            name='Individual Assets',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=asset_names,
            hovertemplate='%{text}<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add minimum variance portfolio
        min_vol = results['min_volatility_portfolio']
        fig.add_trace(go.Scatter(
            x=[min_vol['volatility'] * 100],
            y=[min_vol['return'] * 100],
            mode='markers',
            name='Markowitz Min Vol',
            marker=dict(size=15, color='green', symbol='star'),
            hovertemplate='Markowitz Min Vol<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f"{min_vol['sharpe_ratio']:.3f}" + '<extra></extra>'
        ))
        
        # Add maximum Sharpe ratio portfolio (tangency portfolio)
        max_sharpe = results['max_sharpe_portfolio']
        fig.add_trace(go.Scatter(
            x=[max_sharpe['volatility'] * 100],
            y=[max_sharpe['return'] * 100],
            mode='markers',
            name='Markowitz Max Sharpe',
            marker=dict(size=15, color='gold', symbol='star'),
            hovertemplate='Markowitz Tangency<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f"{max_sharpe['sharpe_ratio']:.3f}" + '<extra></extra>'
        ))
        
        # Add CAAMP optimal portfolio if available
        if caamp_results and caamp_results['optimal_portfolio']:
            caamp_optimal = caamp_results['optimal_portfolio']
            fig.add_trace(go.Scatter(
                x=[caamp_optimal['volatility'] * 100],
                y=[caamp_optimal['return'] * 100],
                mode='markers',
                name='CAAMP Max Sharpe',
                marker=dict(size=15, color='orange', symbol='square'),
                hovertemplate='CAAMP Optimal<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f"{caamp_optimal['sharpe_ratio']:.3f}" + '<extra></extra>'
            ))
        
        # Add Capital Allocation Line (CAL) through max Sharpe portfolio as per efficient.md
        max_vol = max(max(volatilities), max(individual_volatilities)) * 1.2
        cal_vols = np.linspace(0, max_vol, 100)
        cal_returns = risk_free_rate * 100 + max_sharpe['sharpe_ratio'] * cal_vols
        
        fig.add_trace(go.Scatter(
            x=cal_vols,
            y=cal_returns,
            mode='lines',
            name='Capital Allocation Line',
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='CAL<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add risk-free asset
        fig.add_trace(go.Scatter(
            x=[0],
            y=[risk_free_rate * 100],
            mode='markers',
            name='Risk-Free Asset',
            marker=dict(size=12, color='purple', symbol='circle'),
            hovertemplate='Risk-Free Asset<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Dynamic title based on content
        if caamp_results and caamp_results['efficient_portfolios']:
            title = 'Efficient Frontiers: Markowitz vs CAAMP Risk Models'
        else:
            title = 'Markowitz Efficient Frontier - Modern Portfolio Theory'
        
        fig.update_layout(
            title=title,
            xaxis_title='Risk (Annual Volatility %)',
            yaxis_title='Expected Return (Annual %)',
            height=600,
            showlegend=True,
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def calculate_stock_betas(tickers_list, start_date, end_date):
    """Calculate beta for each stock relative to S&P 500 market index"""
    try:
        # Get market benchmark data (S&P 500)
        market_ticker = "^GSPC"  # S&P 500 index
        
        # Fetch market data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            market_data = yf.download(market_ticker, start=start_date, end=end_date, interval="1d", progress=False)
        
        if market_data.empty:
            return None, "Failed to fetch S&P 500 market data for beta calculation"
        
        # Get market returns (monthly)
        market_monthly = market_data['Close'].resample('ME').last()
        market_returns = market_monthly.pct_change().dropna()
        
        if len(market_returns) < 12:
            return None, f"Insufficient market data: only {len(market_returns)} months available"
        
        # Get stock data using existing function
        price_data, status = get_historical_data_custom_range(tickers_list, start_date, end_date)
        if status != "success":
            return None, f"Failed to fetch stock data for beta calculation: {status}"
        
        # Calculate monthly returns for stocks
        stock_monthly = price_data.resample('ME').last()
        stock_returns = stock_monthly.pct_change().dropna()
        
        # Align dates between market and stock returns
        common_dates = market_returns.index.intersection(stock_returns.index)
        aligned_market = market_returns.loc[common_dates]
        aligned_stocks = stock_returns.loc[common_dates]
        
        if len(common_dates) < 12:
            return None, f"Insufficient overlapping data: only {len(common_dates)} months available for beta calculation"
        
        # Calculate beta for each stock
        beta_data = {}
        successful_calculations = 0
        
        for ticker in tickers_list:
            if ticker in aligned_stocks.columns:
                try:
                    stock_data = aligned_stocks[ticker].dropna()
                    market_data_aligned = aligned_market.loc[stock_data.index].dropna()
                    
                    # Ensure we have matching data
                    if len(stock_data) < 12 or len(market_data_aligned) < 12:
                        continue
                    
                    # Calculate covariance and beta using numpy for better control
                    # Ensure both series are aligned and same length
                    common_idx = stock_data.index.intersection(market_data_aligned.index)
                    if len(common_idx) < 12:
                        continue
                    
                    stock_returns_clean = stock_data.loc[common_idx].values
                    market_returns_clean = market_data_aligned.loc[common_idx].values
                    
                    # Calculate beta using pandas methods (more reliable)
                    stock_series = pd.Series(stock_returns_clean.flatten(), index=common_idx)
                    market_series = pd.Series(market_returns_clean.flatten(), index=common_idx)
                    
                    # Calculate covariance and variance
                    covariance = stock_series.cov(market_series)
                    market_variance = market_series.var()
                    beta = covariance / market_variance if market_variance != 0 else 1.0
                    
                    # Calculate correlation
                    correlation = stock_series.corr(market_series)
                    r_squared = correlation ** 2 if not pd.isna(correlation) else 0
                    
                    beta_data[ticker] = {
                        'beta': float(beta),
                        'correlation': float(correlation) if not pd.isna(correlation) else 0,
                        'r_squared': float(r_squared) if not pd.isna(r_squared) else 0
                    }
                    successful_calculations += 1
                    
                except Exception as e:
                    # Log specific ticker errors but continue
                    print(f"Beta calculation failed for {ticker}: {e}")
                    continue
        
        if successful_calculations > 0:
            return beta_data, "success"
        else:
            return None, f"Beta calculations failed for all {len(tickers_list)} stocks"
        
    except Exception as e:
        return None, f"Error in beta calculation function: {str(e)}"

def calculate_portfolio_beta(weights, beta_data, tickers_list):
    """Calculate portfolio beta as weighted average of individual stock betas"""
    try:
        portfolio_beta = 0
        valid_weights = 0
        
        for i, ticker in enumerate(tickers_list):
            if ticker in beta_data and i < len(weights):
                portfolio_beta += weights[i] * beta_data[ticker]['beta']
                valid_weights += weights[i]
        
        # Normalize if some stocks don't have beta data
        if valid_weights > 0:
            portfolio_beta = portfolio_beta / valid_weights
        
        return portfolio_beta
        
    except Exception as e:
        return 1.0  # Default to market beta

def run_monte_carlo_simulation(portfolio_weights, annual_returns, cov_matrix, 
                               initial_investment=100000, time_horizon=10, 
                               num_simulations=10000, confidence_levels=[0.05, 0.95]):
    """
    Run Monte Carlo simulation for portfolio performance
    
    Parameters:
    - portfolio_weights: Portfolio allocation weights
    - annual_returns: Expected annual returns for each asset
    - cov_matrix: Annual covariance matrix
    - initial_investment: Starting portfolio value
    - time_horizon: Investment period in years
    - num_simulations: Number of simulation paths
    - confidence_levels: Confidence intervals to calculate
    
    Returns:
    - Dictionary with simulation results and statistics
    """
    try:
        # Ensure portfolio_weights is a numpy array
        portfolio_weights = np.array(portfolio_weights)
        
        # Portfolio expected return and volatility
        portfolio_return = np.dot(portfolio_weights, annual_returns)
        portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights)))
        
        # Generate random returns for each simulation path
        np.random.seed(42)  # For reproducible results
        random_returns = np.random.normal(
            portfolio_return, 
            portfolio_volatility, 
            (num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns for each path
        simulation_paths = []
        final_values = []
        
        for sim in range(num_simulations):
            path_values = [initial_investment]
            
            for year in range(time_horizon):
                annual_return = random_returns[sim, year]
                new_value = path_values[-1] * (1 + annual_return)
                path_values.append(new_value)
            
            simulation_paths.append(path_values)
            final_values.append(path_values[-1])
        
        # Convert to numpy arrays for easier calculations
        simulation_paths = np.array(simulation_paths)
        final_values = np.array(final_values)
        
        # Calculate statistics
        final_returns = (final_values - initial_investment) / initial_investment
        
        # Percentile calculations
        percentiles = {}
        for conf in confidence_levels:
            percentiles[f'{conf*100:.0f}th'] = pd.Series(final_values).quantile(conf)
        
        # Additional statistics
        mean_final_value = pd.Series(final_values).mean()
        median_final_value = pd.Series(final_values).median()
        std_final_value = pd.Series(final_values).std()
        
        # Probability of loss
        prob_loss = np.sum(final_values < initial_investment) / num_simulations
        
        # Best and worst case scenarios
        best_case = np.max(final_values)
        worst_case = np.min(final_values)
        
        # Annual statistics over the time horizon
        yearly_stats = []
        for year in range(1, time_horizon + 1):
            year_values = simulation_paths[:, year]
            yearly_stats.append({
                'year': year,
                'mean': pd.Series(year_values).mean(),
                'median': pd.Series(year_values).median(),
                'percentile_5': pd.Series(year_values).quantile(0.05),
                'percentile_95': pd.Series(year_values).quantile(0.95),
                'prob_loss': (year_values < initial_investment).sum() / num_simulations
            })
        
        # Compound Annual Growth Rate (CAGR) statistics
        cagr_values = (final_values / initial_investment) ** (1/time_horizon) - 1
        mean_cagr = pd.Series(cagr_values).mean()
        median_cagr = pd.Series(cagr_values).median()
        
        results = {
            'simulation_paths': simulation_paths,
            'final_values': final_values,
            'final_returns': final_returns,
            'statistics': {
                'initial_investment': initial_investment,
                'time_horizon': time_horizon,
                'num_simulations': num_simulations,
                'portfolio_expected_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'mean_final_value': mean_final_value,
                'median_final_value': median_final_value,
                'std_final_value': std_final_value,
                'best_case': best_case,
                'worst_case': worst_case,
                'prob_loss': prob_loss,
                'percentiles': percentiles,
                'mean_cagr': mean_cagr,
                'median_cagr': median_cagr,
                'yearly_stats': yearly_stats
            }
        }
        
        return results, "success"
        
    except Exception as e:
        return None, f"Error in Monte Carlo simulation: {str(e)}"

def create_monte_carlo_plot(mc_results, portfolio_name="Portfolio"):
    """Create Monte Carlo simulation visualization"""
    try:
        simulation_paths = mc_results['simulation_paths']
        mc_stats = mc_results['statistics']
        time_horizon = mc_stats['time_horizon']
        initial_investment = mc_stats['initial_investment']
        
        # Create time axis
        years = list(range(time_horizon + 1))
        
        fig = go.Figure()
        
        # Add a sample of simulation paths (max 100 for performance)
        sample_size = min(100, len(simulation_paths))
        sample_indices = np.random.choice(len(simulation_paths), sample_size, replace=False)
        
        for i in sample_indices:
            fig.add_trace(go.Scatter(
                x=years,
                y=simulation_paths[i],
                mode='lines',
                line=dict(color='lightblue', width=0.5),
                opacity=0.3,
                showlegend=False,
                hovertemplate='Year %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ))
        
        # Add percentile bands
        percentile_5 = []
        percentile_95 = []
        median_path = []
        mean_path = []
        
        for year in range(time_horizon + 1):
            year_values = simulation_paths[:, year]
            percentile_5.append(pd.Series(year_values).quantile(0.05))
            percentile_95.append(pd.Series(year_values).quantile(0.95))
            median_path.append(pd.Series(year_values).median())
            mean_path.append(pd.Series(year_values).mean())
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=percentile_95 + percentile_5[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Band',
            hoverinfo='skip'
        ))
        
        # Add median path
        fig.add_trace(go.Scatter(
            x=years,
            y=median_path,
            mode='lines',
            line=dict(color='darkgreen', width=3),
            name='Median Path',
            hovertemplate='Year %{x}<br>Median Value: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add mean path
        fig.add_trace(go.Scatter(
            x=years,
            y=mean_path,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Mean Path',
            hovertemplate='Year %{x}<br>Mean Value: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add initial investment line
        fig.add_hline(
            y=initial_investment,
            line_dash="dot",
            line_color="black",
            annotation_text="Initial Investment"
        )
        
        fig.update_layout(
            title=f'Monte Carlo Simulation: {portfolio_name}<br><sub>{mc_stats["num_simulations"]:,} simulations over {time_horizon} years</sub>',
            xaxis_title='Years',
            yaxis_title='Portfolio Value ($)',
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Monte Carlo plot: {str(e)}")
        return None

def create_monte_carlo_distribution_plot(mc_results, portfolio_name="Portfolio"):
    """Create histogram of final portfolio values"""
    try:
        final_values = mc_results['final_values']
        mc_stats = mc_results['statistics']
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            name='Final Values',
            opacity=0.7,
            hovertemplate='Value Range: $%{x:,.0f}<br>Frequency: %{y}<extra></extra>'
        ))
        
        # Add vertical lines for key statistics
        fig.add_vline(
            x=mc_stats['mean_final_value'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mc_stats['mean_final_value']:,.0f}"
        )
        
        fig.add_vline(
            x=mc_stats['median_final_value'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: ${mc_stats['median_final_value']:,.0f}"
        )
        
        fig.add_vline(
            x=mc_stats['initial_investment'],
            line_dash="dot",
            line_color="black",
            annotation_text=f"Initial: ${mc_stats['initial_investment']:,.0f}"
        )
        
        # Add percentile markers
        for percentile_name, value in mc_stats['percentiles'].items():
            fig.add_vline(
                x=value,
                line_dash="dashdot",
                line_color="orange",
                annotation_text=f"{percentile_name}: ${value:,.0f}"
            )
        
        fig.update_layout(
            title=f'Distribution of Final Portfolio Values: {portfolio_name}<br><sub>After {mc_stats["time_horizon"]} years ({mc_stats["num_simulations"]:,} simulations)</sub>',
            xaxis_title='Final Portfolio Value ($)',
            yaxis_title='Frequency',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating distribution plot: {str(e)}")
        return None

def create_monte_carlo_statistics_table(mc_results, portfolio_name="Portfolio"):
    """Create statistics table for Monte Carlo results"""
    try:
        mc_stats = mc_results['statistics']
        
        # Format the statistics
        table_data = {
            'Metric': [
                'Expected Annual Return',
                'Annual Volatility',
                'Time Horizon (Years)',
                'Number of Simulations',
                'Initial Investment',
                'Mean Final Value',
                'Median Final Value',
                'Best Case Scenario',
                'Worst Case Scenario',
                '5th Percentile',
                '95th Percentile',
                'Standard Deviation',
                'Probability of Loss',
                'Mean CAGR',
                'Median CAGR'
            ],
            'Value': [
                f"{mc_stats['portfolio_expected_return']*100:.2f}%",
                f"{mc_stats['portfolio_volatility']*100:.2f}%",
                f"{mc_stats['time_horizon']} years",
                f"{mc_stats['num_simulations']:,}",
                f"${mc_stats['initial_investment']:,.0f}",
                f"${mc_stats['mean_final_value']:,.0f}",
                f"${mc_stats['median_final_value']:,.0f}",
                f"${mc_stats['best_case']:,.0f}",
                f"${mc_stats['worst_case']:,.0f}",
                f"${mc_stats['percentiles']['5th']:,.0f}",
                f"${mc_stats['percentiles']['95th']:,.0f}",
                f"${mc_stats['std_final_value']:,.0f}",
                f"{mc_stats['prob_loss']*100:.2f}%",
                f"{mc_stats['mean_cagr']*100:.2f}%",
                f"{mc_stats['median_cagr']*100:.2f}%"
            ]
        }
        
        return pd.DataFrame(table_data)
        
    except Exception as e:
        st.error(f"Error creating statistics table: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header"><h1>📈 Portfolio Performance Analyzer</h1><p>Comprehensive analysis comparing your portfolio with SPX and SPXE</p></div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading portfolio data..."):
        portfolio_data = load_portfolio_data()
    
    if portfolio_data is None:
        st.error("Failed to load portfolio data. Please check if the CSV file exists.")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")
    
    # Date range selector - Allow any reasonable date range
    # Yahoo Finance has reliable data going back to 1970s for major indices
    min_date = pd.to_datetime('1970-01-01').date()  # Practical historical limit for reliable data
    max_date = pd.to_datetime('today').date()  # Current date
    
    st.sidebar.subheader("Date Range Selection")
    st.sidebar.info("📅 Select any date range - S&P 500 data available from **1970** to today!")
    
    # Default to last 5 years for better performance
    default_start = pd.to_datetime('today') - pd.DateOffset(years=5)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start.date(),
        min_value=min_date,
        max_value=max_date,
        help="Yahoo Finance provides S&P 500 data back to 1927!"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Analysis end date (up to today)"
    )
    
    # Validate date range
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        return
    
    # Check if date range is reasonable for analysis
    days_diff = (end_date - start_date).days
    if days_diff < 365:
        st.warning("⚠️ Date range less than 1 year may not provide reliable analysis results.")
    
    # Load market data based on selected date range
    with st.spinner("Loading market data for selected date range..."):
        # Try dynamic data first, fallback to static data if needed
        spx_data = get_spx_data_dynamic(start_date, end_date)
        if spx_data is None or len(spx_data) < 12:
            st.warning("⚠️ Using fallback SPX data - dynamic fetch failed or insufficient data")
            spx_data = get_spx_monthly_data()
        else:
            st.success(f"✅ Loaded {len(spx_data)} months of SPX data from Yahoo Finance")
        
        spxe_data = get_spxe_data_dynamic(start_date, end_date)
        if spxe_data is None or len(spxe_data) < 12:
            st.warning("⚠️ Using fallback SPXE data - dynamic fetch failed or insufficient data")
            spxe_data = get_spxe_monthly_data()
        else:
            st.success(f"✅ Loaded {len(spxe_data)} months of SPXE/RSP data from Yahoo Finance")
    
    # Risk-free rate input
    st.sidebar.subheader("Risk Parameters")
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (Annual %)",
        min_value=0.0,
        max_value=10.0,
        value=3.044,
        step=0.01,
        format="%.3f"
    ) / 100  # Convert to decimal
    
    # Return type selector
    show_total_returns = st.sidebar.selectbox(
        "Return Type",
        ["Annualized Returns", "Total Returns"],
        index=0
    ) == "Total Returns"
    
    # Calculate performance metrics
    portfolio_analysis = calculate_portfolio_returns(portfolio_data, start_date, end_date)
    spx_analysis = calculate_index_returns(spx_data, start_date, end_date)
    spxe_analysis = calculate_index_returns(spxe_data, start_date, end_date)
    
    if portfolio_analysis is None:
        st.error("Please select a date range with at least 2 months of data.")
        return
    
    # Calculate Sharpe ratios
    portfolio_sharpe = calculate_sharpe_ratio(
        portfolio_analysis['annualizedReturn'], 
        portfolio_analysis['annualizedStdDev'], 
        risk_free_rate
    )
    spx_sharpe = calculate_sharpe_ratio(
        spx_analysis['annualizedReturn'] if spx_analysis else 0, 
        spx_analysis['annualizedStdDev'] if spx_analysis else 1, 
        risk_free_rate
    ) if spx_analysis else 0
    spxe_sharpe = calculate_sharpe_ratio(
        spxe_analysis['annualizedReturn'] if spxe_analysis else 0, 
        spxe_analysis['annualizedStdDev'] if spxe_analysis else 1, 
        risk_free_rate
    ) if spxe_analysis else 0
    
    # Performance Comparison Table
    st.subheader("📊 Performance Comparison")
    
    comparison_data = {
        'Metric': [
            'Total Return' if show_total_returns else 'Annualized Return',
            'Annualized Volatility',
            'Sharpe Ratio',
            'Total Months'
        ],
        'Your Portfolio': [
            f"{(portfolio_analysis['totalReturn'] if show_total_returns else portfolio_analysis['annualizedReturn']) * 100:.2f}%",
            f"{portfolio_analysis['annualizedStdDev'] * 100:.2f}%",
            f"{portfolio_sharpe:.3f}",
            str(portfolio_analysis['totalMonths'])
        ],
        'SPX (S&P 500)': [
            f"{(spx_analysis['totalReturn'] if show_total_returns else spx_analysis['annualizedReturn']) * 100:.2f}%" if spx_analysis else 'N/A',
            f"{spx_analysis['annualizedStdDev'] * 100:.2f}%" if spx_analysis else 'N/A',
            f"{spx_sharpe:.3f}" if spx_analysis else 'N/A',
            str(spx_analysis['totalMonths']) if spx_analysis else 'N/A'
        ],
        'SPXE (Equal Weight)': [
            f"{(spxe_analysis['totalReturn'] if show_total_returns else spxe_analysis['annualizedReturn']) * 100:.2f}%" if spxe_analysis else 'N/A',
            f"{spxe_analysis['annualizedStdDev'] * 100:.2f}%" if spxe_analysis else 'N/A',
            f"{spxe_sharpe:.3f}" if spxe_analysis else 'N/A',
            str(spxe_analysis['totalMonths']) if spxe_analysis else 'N/A'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Performance Rankings
    st.subheader(f"🏆 Performance Rankings ({'Total Return' if show_total_returns else 'Sharpe Ratio'})")
    
    portfolios = [
        {
            'name': 'Your Portfolio', 
            'value': portfolio_analysis['totalReturn'] if show_total_returns else portfolio_sharpe,
            'type': 'portfolio'
        }
    ]
    
    if spx_analysis:
        portfolios.append({
            'name': 'SPX (S&P 500)',
            'value': spx_analysis['totalReturn'] if show_total_returns else spx_sharpe,
            'type': 'spx'
        })
    
    if spxe_analysis:
        portfolios.append({
            'name': 'SPXE (Equal Weight)',
            'value': spxe_analysis['totalReturn'] if show_total_returns else spxe_sharpe,
            'type': 'spxe'
        })
    
    portfolios.sort(key=lambda x: x['value'], reverse=True)
    
    for i, p in enumerate(portfolios):
        if i == 0:
            st.success(f"🥇 {i+1}. {p['name']}: {p['value']*100:.2f}%" if show_total_returns else f"🥇 {i+1}. {p['name']}: {p['value']:.3f}")
        elif i == len(portfolios) - 1:
            st.error(f"🥉 {i+1}. {p['name']}: {p['value']*100:.2f}%" if show_total_returns else f"🥉 {i+1}. {p['name']}: {p['value']:.3f}")
        else:
            st.info(f"🥈 {i+1}. {p['name']}: {p['value']*100:.2f}%" if show_total_returns else f"🥈 {i+1}. {p['name']}: {p['value']:.3f}")
    
    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"Your {'Total' if show_total_returns else 'Annualized'} Return",
            f"{(portfolio_analysis['totalReturn'] if show_total_returns else portfolio_analysis['annualizedReturn']) * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Your Volatility",
            f"{portfolio_analysis['annualizedStdDev'] * 100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Your Sharpe Ratio",
            f"{portfolio_sharpe:.3f}"
        )
    
    with col4:
        st.metric(
            "Total Months",
            str(portfolio_analysis['totalMonths'])
        )

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Monthly Returns", 
        "📊 Capital Market Line", 
        "📋 Detailed Statistics", 
        "📂 Portfolio Holdings",
        "🔍 Individual Stock Analysis",
        "💰 Stock Valuation",
        "⚡ Efficient Frontier"
    ])

    with tab1:
        st.subheader("Monthly Returns Comparison")
        
        # Create aligned monthly returns data
        max_length = max(
            len(portfolio_analysis['monthlyReturns']),
            len(spx_analysis['monthlyReturns']) if spx_analysis else 0,
            len(spxe_analysis['monthlyReturns']) if spxe_analysis else 0
        )
        
        aligned_data = []
        for i in range(max_length):
            portfolio_return = portfolio_analysis['monthlyReturns'][i] if i < len(portfolio_analysis['monthlyReturns']) else None
            spx_return = spx_analysis['monthlyReturns'][i] if spx_analysis and i < len(spx_analysis['monthlyReturns']) else None
            spxe_return = spxe_analysis['monthlyReturns'][i] if spxe_analysis and i < len(spxe_analysis['monthlyReturns']) else None
            
            if portfolio_return or spx_return or spxe_return:
                date_val = (portfolio_return['date'] if portfolio_return else 
                           spx_return['date'] if spx_return else 
                           spxe_return['date'])
                
                aligned_data.append({
                    'Date': date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else '',
                    'Your Portfolio': f"{portfolio_return['returnPct']:.2f}%" if portfolio_return else 'N/A',
                    'SPX': f"{spx_return['returnPct']:.2f}%" if spx_return else 'N/A',
                    'SPXE': f"{spxe_return['returnPct']:.2f}%" if spxe_return else 'N/A'
                })
        
        returns_df = pd.DataFrame(aligned_data)
        st.dataframe(returns_df, use_container_width=True, hide_index=True)
        
        # Plot monthly returns
        import plotly.graph_objects as go  # Explicit import to avoid scope issues
        fig_returns = go.Figure()
        
        # Portfolio returns
        portfolio_dates = [r['date'] for r in portfolio_analysis['monthlyReturns']]
        portfolio_returns = [r['returnPct'] for r in portfolio_analysis['monthlyReturns']]
        
        fig_returns.add_trace(go.Scatter(
            x=portfolio_dates,
            y=portfolio_returns,
            mode='lines+markers',
            name='Your Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        # SPX returns
        if spx_analysis:
            spx_dates = [r['date'] for r in spx_analysis['monthlyReturns']]
            spx_returns = [r['returnPct'] for r in spx_analysis['monthlyReturns']]
            
            fig_returns.add_trace(go.Scatter(
                x=spx_dates,
                y=spx_returns,
                mode='lines+markers',
                name='SPX',
                line=dict(color='red', width=2)
            ))
        
        # SPXE returns
        if spxe_analysis:
            spxe_dates = [r['date'] for r in spxe_analysis['monthlyReturns']]
            spxe_returns = [r['returnPct'] for r in spxe_analysis['monthlyReturns']]
            
            fig_returns.add_trace(go.Scatter(
                x=spxe_dates,
                y=spxe_returns,
                mode='lines+markers',
                name='SPXE',
                line=dict(color='green', width=2)
            ))
        
        fig_returns.update_layout(
            title='Monthly Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Monthly Return (%)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)

    with tab2:
        st.subheader("Capital Market Line Analysis")

        # Create and display the CML plot
        cml_fig = create_cml_plot(
            portfolio_analysis['annualizedReturn'], 
            portfolio_analysis['annualizedStdDev'],
            spx_analysis['annualizedReturn'] if spx_analysis else 0.1,
            spx_analysis['annualizedStdDev'] if spx_analysis else 0.15,
            risk_free_rate
        )
        
        # Add SPXE point if available
        if spxe_analysis:
            cml_fig.add_trace(go.Scatter(
                x=[spxe_analysis['annualizedStdDev'] * 100],
                y=[spxe_analysis['annualizedReturn'] * 100],
                mode='markers',
                name='SPXE (Equal Weight)',
                marker=dict(color='green', size=12, symbol='triangle-up')
            ))
        
        st.plotly_chart(cml_fig, use_container_width=True)

        # Explanation and insights
        st.markdown("### 📖 Understanding the Capital Market Line")
        
        market_sharpe = calculate_sharpe_ratio(
            spx_analysis['annualizedReturn'] if spx_analysis else 0.1,
            spx_analysis['annualizedStdDev'] if spx_analysis else 0.15,
            risk_free_rate
        )
        
        st.markdown(f"""
        The **Capital Market Line (CML)** represents the optimal risk-return trade-off available to investors using:
        - The risk-free asset (earning {risk_free_rate*100:.3f}% annually)
        - The market portfolio (SPX)
        - Combinations of both
        
        **Key Insights:**
        - **Market Sharpe Ratio**: {market_sharpe:.3f} (slope of the CML)
        - **Your Portfolio**: Risk = {portfolio_analysis['annualizedStdDev']*100:.2f}%, Return = {portfolio_analysis['annualizedReturn']*100:.2f}%
        - **Market Portfolio (SPX)**: Risk = {spx_analysis['annualizedStdDev']*100:.2f}%, Return = {spx_analysis['annualizedReturn']*100:.2f}%
        
        **Performance Analysis:**
        - Portfolios **above** the CML offer superior risk-adjusted returns
        - Portfolios **below** the CML are sub-optimal (you could get better returns for the same risk)
        - The CML represents the **efficient frontier** when combining risk-free assets with the market
        """)

        # Portfolio efficiency analysis
        portfolio_expected_return_on_cml = risk_free_rate + market_sharpe * portfolio_analysis['annualizedStdDev']
        alpha = portfolio_analysis['annualizedReturn'] - portfolio_expected_return_on_cml
        
        if alpha > 0:
            st.success(f"🎯 **Your portfolio is ABOVE the CML!** Alpha = {alpha*100:.2f}%")
            st.success("Your portfolio is generating superior risk-adjusted returns compared to the optimal market combination.")
        elif alpha < -0.005:  # Allow small margin for rounding
            st.error(f"📉 **Your portfolio is BELOW the CML.** Alpha = {alpha*100:.2f}%")
            st.error("You could achieve better returns for the same risk by combining SPX with risk-free assets.")
        else:
            st.info(f"⚖️ **Your portfolio is ON the CML.** Alpha = {alpha*100:.2f}%")
            st.info("Your portfolio is efficiently positioned on the capital market line.")

        # Optimal allocation examples
        st.markdown("### 🎯 Optimal Portfolio Allocations on the CML")
        allocations = [0.25, 0.5, 0.75, 1.0, 1.25]
        allocation_data = []

        for alloc in allocations:
            portfolio_return = risk_free_rate + alloc * (spx_analysis['annualizedReturn'] - risk_free_rate) if spx_analysis else risk_free_rate
            portfolio_risk = alloc * (spx_analysis['annualizedStdDev'] if spx_analysis else 0.15)
            leverage_text = "Leveraged" if alloc > 1 else "Standard"
            
            allocation_data.append({
                'SPX Allocation': f"{alloc*100:.0f}%",
                'Risk-Free Allocation': f"{(1-alloc)*100:.0f}%",
                'Type': leverage_text,
                'Expected Return': f"{portfolio_return*100:.2f}%",
                'Risk (Volatility)': f"{portfolio_risk*100:.2f}%",
                'Sharpe Ratio': f"{market_sharpe:.3f}"
            })

        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df, use_container_width=True)

    with tab3:
        st.subheader("Detailed Summary Statistics")

        # Create comprehensive statistics table
        detailed_stats = {
            'Metric': [
                'Start Date', 'End Date', 'Number of Periods',
                'Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
                'Monthly Mean Return', 'Monthly Std Deviation',
                'Risk-Free Rate'
            ],
            'Your Portfolio': [
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                str(portfolio_analysis['totalMonths']),
                f"{portfolio_analysis['totalReturn']*100:.2f}%",
                f"{portfolio_analysis['annualizedReturn']*100:.2f}%",
                f"{portfolio_analysis['annualizedStdDev']*100:.2f}%",
                f"{portfolio_sharpe:.3f}",
                f"{portfolio_analysis['meanReturn']*100:.3f}%",
                f"{portfolio_analysis['monthlyStdDev']*100:.3f}%",
                f"{risk_free_rate*100:.3f}%"
            ],
            'SPX': [
                start_date.strftime('%Y-%m-%d') if spx_analysis else 'N/A',
                end_date.strftime('%Y-%m-%d') if spx_analysis else 'N/A',
                str(spx_analysis['totalMonths']) if spx_analysis else 'N/A',
                f"{spx_analysis['totalReturn']*100:.2f}%" if spx_analysis else 'N/A',
                f"{spx_analysis['annualizedReturn']*100:.2f}%" if spx_analysis else 'N/A',
                f"{spx_analysis['annualizedStdDev']*100:.2f}%" if spx_analysis else 'N/A',
                f"{spx_sharpe:.3f}" if spx_analysis else 'N/A',
                f"{spx_analysis['meanReturn']*100:.3f}%" if spx_analysis else 'N/A',
                f"{spx_analysis['monthlyStdDev']*100:.3f}%" if spx_analysis else 'N/A',
                f"{risk_free_rate*100:.3f}%" if spx_analysis else 'N/A'
            ],
            'SPXE': [
                start_date.strftime('%Y-%m-%d') if spxe_analysis else 'N/A',
                end_date.strftime('%Y-%m-%d') if spxe_analysis else 'N/A',
                str(spxe_analysis['totalMonths']) if spxe_analysis else 'N/A',
                f"{spxe_analysis['totalReturn']*100:.2f}%" if spxe_analysis else 'N/A',
                f"{spxe_analysis['annualizedReturn']*100:.2f}%" if spxe_analysis else 'N/A',
                f"{spxe_analysis['annualizedStdDev']*100:.2f}%" if spxe_analysis else 'N/A',
                f"{spxe_sharpe:.3f}" if spxe_analysis else 'N/A',
                f"{spxe_analysis['meanReturn']*100:.3f}%" if spxe_analysis else 'N/A',
                f"{spxe_analysis['monthlyStdDev']*100:.3f}%" if spxe_analysis else 'N/A',
                f"{risk_free_rate*100:.3f}%" if spxe_analysis else 'N/A'
            ]
        }

        detailed_df = pd.DataFrame(detailed_stats)
        st.dataframe(detailed_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("📂 Current Portfolio Holdings")
        
        # Load portfolio positions
        portfolio_positions = load_portfolio_positions()
        
        if portfolio_positions is not None and len(portfolio_positions) > 0:
            st.success(f"✅ Successfully loaded {len(portfolio_positions)} portfolio positions")
            
            # Automatically fetch current market prices and update values
            with st.spinner("Fetching current market prices..."):
                try:
                    updated_positions = portfolio_positions.copy()
                    successful_updates = 0
                    failed_updates = []
                    
                    for idx, position in updated_positions.iterrows():
                        symbol = position['Symbol']
                        
                        try:
                            # Fetch current price from Yahoo Finance
                            ticker = yf.Ticker(symbol)
                            info = ticker.info
                            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
                            
                            if current_price and current_price > 0:
                                # Update the price and calculate new values
                                quantity = position['Quantity']
                                
                                # Calculate new current value
                                new_current_value = current_price * quantity
                                
                                # Calculate new gain/loss (assuming cost basis remains the same)
                                cost_basis = position['Current_Value'] - position['Total_Gain_Loss_Dollar']
                                new_gain_loss_dollar = new_current_value - cost_basis
                                new_gain_loss_percent = new_gain_loss_dollar / cost_basis if cost_basis != 0 else 0
                                
                                # Update the dataframe
                                updated_positions.loc[idx, 'Last_Price'] = current_price
                                updated_positions.loc[idx, 'Current_Value'] = new_current_value
                                updated_positions.loc[idx, 'Total_Gain_Loss_Dollar'] = new_gain_loss_dollar
                                updated_positions.loc[idx, 'Total_Gain_Loss_Percent'] = new_gain_loss_percent
                                
                                successful_updates += 1
                            else:
                                failed_updates.append(symbol)
                                
                        except Exception as e:
                            failed_updates.append(f"{symbol}")
                            continue
                    
                    # Use updated values for all calculations
                    portfolio_positions = updated_positions
                    
                    # Calculate updated portfolio totals
                    total_value = portfolio_positions['Current_Value'].sum()
                    total_gain_loss = portfolio_positions['Total_Gain_Loss_Dollar'].sum()
                    weighted_gain_loss_pct = (total_gain_loss / (total_value - total_gain_loss)) * 100
                    
                    # Recalculate portfolio weights
                    portfolio_positions['Percent_Of_Account'] = portfolio_positions['Current_Value'] / total_value
                    
                    # Show update status
                    if successful_updates > 0:
                        st.success(f"✅ Real-time prices updated for {successful_updates} positions at {pd.Timestamp.now().strftime('%H:%M:%S')}")
                    if failed_updates:
                        st.warning(f"⚠️ Could not update prices for {len(failed_updates)} positions: {', '.join(failed_updates[:5])}")
                        
                except Exception as e:
                    st.warning(f"⚠️ Some prices may not be current due to market data limitations")
                    # Fall back to original values if update fails completely
                    total_value = portfolio_positions['Current_Value'].sum()
                    total_gain_loss = portfolio_positions['Total_Gain_Loss_Dollar'].sum()
                    weighted_gain_loss_pct = (total_gain_loss / (total_value - total_gain_loss)) * 100
            
            # Display current portfolio summary metrics
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}")
            with col3:
                st.metric("Portfolio Return", f"{weighted_gain_loss_pct:.2f}%")
            with col4:
                if st.button("🔄", help="Refresh prices", key="refresh_prices"):
                    st.rerun()
            
            st.markdown("---")
            
            # Format the dataframe for display
            display_df = portfolio_positions.copy()
            display_df['Current_Value'] = display_df['Current_Value'].apply(lambda x: f"${x:,.2f}")
            display_df['Last_Price'] = display_df['Last_Price'].apply(lambda x: f"${x:.2f}")
            display_df['Percent_Of_Account'] = display_df['Percent_Of_Account'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Total_Gain_Loss_Dollar'] = display_df['Total_Gain_Loss_Dollar'].apply(
                lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
            )
            display_df['Total_Gain_Loss_Percent'] = display_df['Total_Gain_Loss_Percent'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Quantity'] = display_df['Quantity'].apply(lambda x: f"{x:.3f}")
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'Symbol': 'Ticker',
                'Description': 'Company Name',
                'Quantity': 'Shares',
                'Last_Price': 'Price',
                'Current_Value': 'Market Value',
                'Percent_Of_Account': '% of Portfolio',
                'Total_Gain_Loss_Dollar': 'Gain/Loss ($)',
                'Total_Gain_Loss_Percent': 'Gain/Loss (%)'
            })
            
            # Display the holdings table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Portfolio composition chart
            st.subheader("📊 Portfolio Composition")
            
            # Create pie chart of top holdings
            top_holdings = portfolio_positions.nlargest(10, 'Current_Value')
            others_value = portfolio_positions['Current_Value'].sum() - top_holdings['Current_Value'].sum()
            
            if others_value > 0:
                # Add "Others" category
                chart_data = list(zip(top_holdings['Symbol'], top_holdings['Current_Value']))
                chart_data.append(('Others', others_value))
                symbols, values = zip(*chart_data)
            else:
                symbols = top_holdings['Symbol']
                values = top_holdings['Current_Value']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=symbols,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig_pie.update_layout(
                title='Portfolio Allocation by Holdings',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Portfolio Beta Analysis
            st.subheader("📊 Portfolio Beta Analysis")
            
            if st.button("🔍 Calculate Portfolio Beta"):
                with st.spinner("Calculating portfolio beta vs S&P 500..."):
                    beta_analysis = get_portfolio_beta_analysis(portfolio_positions)
                
                if beta_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        portfolio_beta = beta_analysis['portfolio_beta']
                        st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
                        
                        # Interpretation
                        if portfolio_beta > 1.2:
                            st.error(f"🔴 High Risk: {((portfolio_beta-1)*100):+.0f}% more volatile than market")
                        elif portfolio_beta > 0.8:
                            st.info(f"🟡 Moderate Risk: {((portfolio_beta-1)*100):+.0f}% vs market volatility")
                        else:
                            st.success(f"🟢 Low Risk: {((1-portfolio_beta)*100):.0f}% less volatile than market")
                    
                    with col2:
                        coverage = beta_analysis['coverage']
                        total_positions = beta_analysis['total_positions']
                        st.metric("Beta Coverage", f"{coverage*100:.0f}% ({total_positions} stocks)")
                    
                    # Individual stock betas
                    st.markdown("**Individual Stock Betas**")
                    beta_df = beta_analysis['individual_betas'].copy()
                    
                    # Format for display
                    beta_df['Weight'] = beta_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
                    beta_df['Beta'] = beta_df['Beta'].apply(lambda x: f"{x:.2f}")
                    beta_df['Correlation'] = beta_df['Correlation'].apply(lambda x: f"{x:.2f}")
                    beta_df['Current_Value'] = beta_df['Current_Value'].apply(lambda x: f"${x:,.0f}")
                    beta_df['Weighted_Beta'] = beta_df['Weighted_Beta'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(
                        beta_df[['Symbol', 'Weight', 'Beta', 'Correlation', 'Current_Value', 'Weighted_Beta']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Portfolio Beta Scatter Plot Visualization
                    st.markdown("**📊 Portfolio vs S&P 500 Returns Scatter Plot**")
                    st.info("This scatter plot visualizes the beta relationship by showing your portfolio's monthly returns vs S&P 500 returns.")
                    
                    # Auto-generate the scatter plot when portfolio beta is calculated
                    with st.spinner("Creating portfolio beta scatter plot..."):
                        try:
                            # Calculate portfolio monthly returns
                            portfolio_data = load_portfolio_data()
                            if portfolio_data is not None:
                                # Use the date range from sidebar
                                portfolio_analysis = calculate_portfolio_returns(portfolio_data, start_date, end_date)
                                
                                if portfolio_analysis and len(portfolio_analysis['monthlyReturns']) > 1:
                                    # Get SPX data for the same period
                                    spx_data = get_spx_data_dynamic(start_date, end_date)
                                    if spx_data is None:
                                        spx_data = get_spx_monthly_data()
                                    
                                    spx_analysis = calculate_index_returns(spx_data, start_date, end_date)
                                    
                                    if spx_analysis and len(spx_analysis['monthlyReturns']) > 1:
                                        # Align the data by dates
                                        portfolio_returns_dict = {pd.to_datetime(r['date']).strftime('%Y-%m'): r['returnPct'] 
                                                                for r in portfolio_analysis['monthlyReturns']}
                                        spx_returns_dict = {pd.to_datetime(r['date']).strftime('%Y-%m'): r['returnPct'] 
                                                          for r in spx_analysis['monthlyReturns']}
                                        
                                        # Find common dates
                                        common_dates = set(portfolio_returns_dict.keys()) & set(spx_returns_dict.keys())
                                        
                                        if len(common_dates) > 1:
                                            # Create aligned lists for scatter plot
                                            portfolio_returns = [portfolio_returns_dict[date] for date in sorted(common_dates)]
                                            spx_returns = [spx_returns_dict[date] for date in sorted(common_dates)]
                                            
                                            # Create scatter plot
                                            fig_scatter = go.Figure()
                                            
                                            # Add scatter points
                                            fig_scatter.add_trace(go.Scatter(
                                                x=spx_returns,
                                                y=portfolio_returns,
                                                mode='markers',
                                                name='Monthly Returns',
                                                marker=dict(
                                                    size=8,
                                                    color='blue',
                                                    opacity=0.7,
                                                    line=dict(width=1, color='darkblue')
                                                ),
                                                text=[f"Date: {date}<br>SPX: {spx:.2f}%<br>Portfolio: {port:.2f}%" 
                                                      for date, spx, port in zip(sorted(common_dates), spx_returns, portfolio_returns)],
                                                hovertemplate='%{text}<extra></extra>'
                                            ))
                                            
                                            # Add beta trendline
                                            slope, intercept, r_value, p_value, std_err = stats.linregress(spx_returns, portfolio_returns)
                                            line_x = np.array([min(spx_returns), max(spx_returns)])
                                            line_y = slope * line_x + intercept
                                            
                                            fig_scatter.add_trace(go.Scatter(
                                                x=line_x,
                                                y=line_y,
                                                mode='lines',
                                                name=f'Beta Trendline (β={slope:.2f})',
                                                line=dict(color='red', width=3, dash='dash')
                                            ))
                                            
                                            # Update layout
                                            fig_scatter.update_layout(
                                                title=f'Portfolio Beta Analysis: β = {slope:.2f}, R² = {r_value**2:.3f}',
                                                xaxis_title='S&P 500 Monthly Returns (%)',
                                                yaxis_title='Portfolio Monthly Returns (%)',
                                                height=500,
                                                showlegend=True,
                                                hovermode='closest'
                                            )
                                            
                                            # Add reference lines at 0
                                            fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                                            fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                                            
                                            st.plotly_chart(fig_scatter, use_container_width=True)
                                            
                                            # Display beta metrics in a professional layout
                                            beta_col1, beta_col2, beta_col3, beta_col4 = st.columns(4)
                                            with beta_col1:
                                                st.metric("Portfolio Beta", f"{slope:.3f}")
                                            with beta_col2:
                                                st.metric("R-squared", f"{r_value**2:.3f}")
                                            with beta_col3:
                                                st.metric("Correlation", f"{r_value:.3f}")
                                            with beta_col4:
                                                st.metric("Data Points", len(common_dates))
                                            
                                            # Enhanced beta interpretation with color coding
                                            if slope > 1.2:
                                                st.error(f"🔴 **High Beta Portfolio**: {((slope-1)*100):+.0f}% more volatile than the market")
                                                st.write("**Interpretation**: Your portfolio tends to amplify market movements - higher potential returns but also higher risk.")
                                            elif slope > 0.8:
                                                st.info(f"🟡 **Moderate Beta Portfolio**: {((slope-1)*100):+.0f}% vs market volatility")
                                                st.write("**Interpretation**: Your portfolio moves roughly in line with the market - balanced risk profile.")
                                            else:
                                                st.success(f"🟢 **Low Beta Portfolio**: {((1-slope)*100):.0f}% less volatile than the market")
                                                st.write("**Interpretation**: Your portfolio is more conservative - lower volatility but potentially lower returns.")
                                        else:
                                            st.error("Insufficient overlapping data points for scatter plot")
                                    else:
                                        st.error("Could not calculate SPX returns for the selected period")
                                else:
                                    st.error("Could not calculate portfolio returns for the selected period")
                            else:
                                st.error("Portfolio data not available")
                        except Exception as e:
                            st.error(f"Error creating scatter plot: {str(e)}")
                    
                else:
                    st.error("Could not calculate portfolio beta. Insufficient data.")
            
        else:
            st.warning("⚠️ No portfolio positions data found. Please ensure the Portfolio_Positions_Aug-11-2025.csv file is available.")
            st.info("💡 The portfolio analysis is currently using aggregated balance data from Investment_income_balance_detail.csv")

    with tab5:
        st.subheader("🔍 Individual Stock Analysis")
        
        # Load portfolio positions for stock selection
        portfolio_positions = load_portfolio_positions()
        
        if portfolio_positions is not None and len(portfolio_positions) > 0:
            stock_symbols = portfolio_positions['Symbol'].tolist()
            
            # Stock selector
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_stock = st.selectbox(
                    "Select a stock from your portfolio:",
                    options=stock_symbols,
                    help="Choose any stock from your current portfolio holdings"
                )
            with col2:
                custom_ticker = st.text_input(
                    "Or enter any ticker:",
                    placeholder="e.g., AAPL",
                    help="Enter any stock ticker for analysis"
                )
            
            # Use custom ticker if provided, otherwise use selected stock
            analysis_ticker = custom_ticker.upper() if custom_ticker else selected_stock
            
            if analysis_ticker:
                st.markdown(f"### Analysis for {analysis_ticker}")
                
                # Show portfolio position info if it's from portfolio
                if analysis_ticker in stock_symbols:
                    position_info = portfolio_positions[portfolio_positions['Symbol'] == analysis_ticker].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Shares Held", f"{position_info['Quantity']:.3f}")
                    with col2:
                        st.metric("Current Price", f"${position_info['Last_Price']:.2f}")
                    with col3:
                        st.metric("Market Value", f"${position_info['Current_Value']:,.2f}")
                    with col4:
                        gain_loss_pct = position_info['Total_Gain_Loss_Percent'] * 100
                        st.metric("Total Return", f"{gain_loss_pct:.2f}%")
                
                # Fetch and display real stock data
                with st.spinner(f"Fetching data for {analysis_ticker}..."):
                    stock_data = fetch_stock_data(analysis_ticker)
                
                if stock_data:
                    # Company overview
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Company Information**")
                        st.write(f"**Name**: {stock_data['name']}")
                        st.write(f"**Current Price**: {format_currency(stock_data['price'])}")
                        st.write(f"**Market Cap**: {format_currency(stock_data['market_cap'])}")
                        st.write(f"**Beta**: {stock_data['beta']:.2f}" if stock_data['beta'] else "Beta: N/A")
                    
                    with col2:
                        st.markdown("**Key Ratios**")
                        st.write(f"**P/E Ratio**: {stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] else "P/E: N/A")
                        st.write(f"**P/B Ratio**: {stock_data['pb_ratio']:.2f}" if stock_data['pb_ratio'] else "P/B: N/A")
                        st.write(f"**P/FCF Ratio**: {stock_data['p_fcf_ratio']:.1f}" if stock_data['p_fcf_ratio'] else "P/FCF: N/A")
                        st.write(f"**Dividend Yield**: {format_dividend_yield(stock_data['dividend_yield'])}")
                    
                    # Financial health metrics
                    st.markdown("**Financial Health**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Free Cash Flow", format_currency(stock_data['free_cash_flow']))
                        st.caption(f"Method: {stock_data['fcf_method']}")
                    with col2:
                        st.metric("ROE", format_percentage(stock_data['roe']))
                    with col3:
                        st.metric("Debt/Equity", f"{stock_data['debt_to_equity']:.2f}" if stock_data['debt_to_equity'] else "N/A")
                    
                    # 52-week range
                    if stock_data['week_52_low'] and stock_data['week_52_high']:
                        current_price = stock_data['price']
                        low_52 = stock_data['week_52_low']
                        high_52 = stock_data['week_52_high']
                        
                        range_position = (current_price - low_52) / (high_52 - low_52) if high_52 != low_52 else 0.5
                        
                        st.markdown("**52-Week Range**")
                        st.progress(range_position)
                        st.write(f"Low: {format_currency(low_52)} | Current: {format_currency(current_price)} | High: {format_currency(high_52)}")
                
                else:
                    st.error(f"Could not fetch data for {analysis_ticker}")
                
                # Portfolio position details if applicable
                if analysis_ticker in stock_symbols:
                    position = portfolio_positions[portfolio_positions['Symbol'] == analysis_ticker].iloc[0]
                    st.markdown("**Your Position Details**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Shares Held", f"{position['Quantity']:.3f}")
                    with col2:
                        st.metric("Portfolio Weight", f"{position['Percent_Of_Account']*100:.2f}%")
                    with col3:
                        gain_loss = position['Total_Gain_Loss_Percent'] * 100
                        st.metric("Total Return", f"{gain_loss:.2f}%")
                
                # Monthly Returns Comparison Table
                st.markdown("**📊 Monthly Returns vs S&P 500**")
                if st.button(f"Generate Monthly Returns Comparison for {analysis_ticker}", key=f"monthly_returns_{analysis_ticker}"):
                    with st.spinner("Calculating monthly returns comparison..."):
                        try:
                            # Get stock and SPX data for the selected date range
                            stock_data_hist, status = get_historical_data_custom_range([analysis_ticker], start_date, end_date)
                            
                            # Get SPX data for comparison
                            spx_data_comparison = get_spx_data_dynamic(start_date, end_date)
                            if spx_data_comparison is None:
                                spx_data_comparison = get_spx_monthly_data()
                            
                            if status == "success" and stock_data_hist is not None and spx_data_comparison is not None:
                                # Calculate monthly returns for both
                                stock_monthly = stock_data_hist[analysis_ticker].resample('ME').last()
                                stock_returns = stock_monthly.pct_change().dropna()
                                
                                spx_monthly = spx_data_comparison.set_index('date')['price'].resample('ME').last()
                                spx_returns = spx_monthly.pct_change().dropna()
                                
                                # Align dates
                                common_dates = stock_returns.index.intersection(spx_returns.index)
                                
                                if len(common_dates) > 0:
                                    # Create comparison dataframe
                                    comparison_df = pd.DataFrame({
                                        'Date': common_dates,
                                        f'{analysis_ticker} Return': stock_returns.loc[common_dates] * 100,
                                        'S&P 500 Return': spx_returns.loc[common_dates] * 100,
                                    })
                                    
                                    # Calculate outperformance
                                    comparison_df['Outperformance'] = comparison_df[f'{analysis_ticker} Return'] - comparison_df['S&P 500 Return']
                                    
                                    # Format dates for display
                                    comparison_df['Date'] = comparison_df['Date'].dt.strftime('%Y-%m')
                                    
                                    # Sort by date (newest first)
                                    comparison_df = comparison_df.sort_values('Date', ascending=False)
                                    
                                    # Format the numerical columns for better display
                                    display_df = comparison_df.copy()
                                    display_df[f'{analysis_ticker} Return'] = display_df[f'{analysis_ticker} Return'].apply(lambda x: f"{x:.2f}%")
                                    display_df['S&P 500 Return'] = display_df['S&P 500 Return'].apply(lambda x: f"{x:.2f}%")
                                    display_df['Outperformance'] = display_df['Outperformance'].apply(lambda x: f"{x:.2f}%")
                                    
                                    # Display the table
                                    st.dataframe(
                                        display_df,
                                        use_container_width=True,
                                        height=400
                                    )
                                    
                                    # Summary statistics
                                    stock_mean = comparison_df[f'{analysis_ticker} Return'].mean()
                                    spx_mean = comparison_df['S&P 500 Return'].mean()
                                    avg_outperformance = comparison_df['Outperformance'].mean()
                                    win_rate = (comparison_df['Outperformance'] > 0).mean() * 100
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Avg Monthly Return", f"{stock_mean:.2f}%")
                                    with col2:
                                        st.metric("S&P 500 Avg", f"{spx_mean:.2f}%")
                                    with col3:
                                        st.metric("Avg Outperformance", f"{avg_outperformance:.2f}%")
                                    with col4:
                                        st.metric("Win Rate", f"{win_rate:.1f}%")
                                    
                                    # Add scatterplot with beta analysis
                                    st.markdown("**📊 Beta Analysis Scatterplot**")
                                    
                                    # Calculate beta and correlation for the stock
                                    stock_returns_for_beta = stock_returns.loc[common_dates].values
                                    spx_returns_for_beta = spx_returns.loc[common_dates].values
                                    
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(spx_returns_for_beta, stock_returns_for_beta)
                                    
                                    # Create scatterplot
                                    fig_scatter = go.Figure()
                                    
                                    # Add scatter points
                                    fig_scatter.add_trace(go.Scatter(
                                        x=spx_returns_for_beta * 100,  # Convert to percentages
                                        y=stock_returns_for_beta * 100,  # Convert to percentages
                                        mode='markers',
                                        name='Monthly Returns',
                                        marker=dict(
                                            size=8,
                                            color='blue',
                                            opacity=0.7,
                                            line=dict(width=1, color='darkblue')
                                        ),
                                        text=[f"Date: {date}<br>SPX: {spx:.2f}%<br>{analysis_ticker}: {stock:.2f}%" 
                                              for date, spx, stock in zip(common_dates.strftime('%Y-%m'), 
                                                                        spx_returns_for_beta * 100, 
                                                                        stock_returns_for_beta * 100)],
                                        hovertemplate='%{text}<extra></extra>'
                                    ))
                                    
                                    # Add beta trendline
                                    line_x = np.array([spx_returns_for_beta.min(), spx_returns_for_beta.max()]) * 100
                                    line_y = (slope * np.array([spx_returns_for_beta.min(), spx_returns_for_beta.max()]) + intercept) * 100
                                    
                                    fig_scatter.add_trace(go.Scatter(
                                        x=line_x,
                                        y=line_y,
                                        mode='lines',
                                        name=f'Beta Trendline (β={slope:.2f})',
                                        line=dict(color='red', width=3, dash='dash')
                                    ))
                                    
                                    # Update layout
                                    fig_scatter.update_layout(
                                        title=f'{analysis_ticker} Beta Analysis: β = {slope:.2f}, R² = {r_value**2:.3f}',
                                        xaxis_title='S&P 500 Monthly Returns (%)',
                                        yaxis_title=f'{analysis_ticker} Monthly Returns (%)',
                                        height=500,
                                        showlegend=True,
                                        hovermode='closest'
                                    )
                                    
                                    # Add reference lines at 0
                                    fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                                    fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                                    
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                    
                                    # Display beta metrics
                                    beta_col1, beta_col2, beta_col3, beta_col4 = st.columns(4)
                                    with beta_col1:
                                        st.metric("Beta", f"{slope:.3f}")
                                    with beta_col2:
                                        st.metric("R-squared", f"{r_value**2:.3f}")
                                    with beta_col3:
                                        st.metric("Correlation", f"{r_value:.3f}")
                                    with beta_col4:
                                        st.metric("Data Points", len(common_dates))
                                    
                                    # Beta interpretation
                                    if slope > 1.2:
                                        st.error(f"🔴 High Beta: {analysis_ticker} is {((slope-1)*100):+.0f}% more volatile than the market")
                                    elif slope > 0.8:
                                        st.info(f"🟡 Moderate Beta: {analysis_ticker} volatility is {((slope-1)*100):+.0f}% vs market")
                                    else:
                                        st.success(f"🟢 Low Beta: {analysis_ticker} is {((1-slope)*100):.0f}% less volatile than the market")
                                        
                                else:
                                    st.error("No overlapping dates found between stock and S&P 500 data")
                            else:
                                st.error(f"Failed to fetch historical data for {analysis_ticker}")
                        except Exception as e:
                            # Log the error but provide a user-friendly message
                            error_msg = str(e)
                            if "background_gradient" in error_msg or "matplotlib" in error_msg:
                                st.error("Error calculating monthly returns: Styling feature not available in this environment.")
                            else:
                                st.error(f"Error calculating monthly returns: Unable to fetch or process data for {analysis_ticker}")
                            st.info("💡 Try using the basic stock information section above instead.")
                
        else:
            st.warning("⚠️ No portfolio data available for stock selection")
            
            # Allow manual ticker entry even without portfolio data
            manual_ticker = st.text_input(
                "Enter any stock ticker for analysis:",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                help="Enter any stock ticker symbol"
            )
            
            if manual_ticker:
                ticker_upper = manual_ticker.upper()
                st.info(f"📊 Analysis for {ticker_upper}")
                
                # Basic stock information
                with st.spinner(f"Fetching data for {ticker_upper}..."):
                    stock_data = fetch_stock_data(ticker_upper)
                
                if stock_data:
                    # Company overview
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Company Information**")
                        st.write(f"**Name**: {stock_data['name']}")
                        st.write(f"**Current Price**: {format_currency(stock_data['price'])}")
                        st.write(f"**Market Cap**: {format_currency(stock_data['market_cap'])}")
                        st.write(f"**Beta**: {stock_data['beta']:.2f}" if stock_data['beta'] else "Beta: N/A")
                    
                    with col2:
                        st.markdown("**Key Ratios**")
                        st.write(f"**P/E Ratio**: {stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] else "P/E: N/A")
                        st.write(f"**P/B Ratio**: {stock_data['pb_ratio']:.2f}" if stock_data['pb_ratio'] else "P/B: N/A")
                        st.write(f"**P/FCF Ratio**: {stock_data['p_fcf_ratio']:.1f}" if stock_data['p_fcf_ratio'] else "P/FCF: N/A")
                        st.write(f"**Dividend Yield**: {format_dividend_yield(stock_data['dividend_yield'])}")
                
                # Monthly Returns Comparison Table for manual entry
                st.markdown("**📊 Monthly Returns vs S&P 500**")
                if st.button(f"Generate Monthly Returns Comparison for {ticker_upper}", key=f"monthly_returns_manual_{ticker_upper}"):
                    with st.spinner("Calculating monthly returns comparison..."):
                        try:
                            # Get stock and SPX data for the selected date range
                            stock_data_hist, status = get_historical_data_custom_range([ticker_upper], start_date, end_date)
                            
                            # Get SPX data for comparison
                            spx_data_comparison = get_spx_data_dynamic(start_date, end_date)
                            if spx_data_comparison is None:
                                spx_data_comparison = get_spx_monthly_data()
                            
                            if status == "success" and stock_data_hist is not None and spx_data_comparison is not None:
                                # Calculate monthly returns for both
                                stock_monthly = stock_data_hist[ticker_upper].resample('ME').last()
                                stock_returns = stock_monthly.pct_change().dropna()
                                
                                spx_monthly = spx_data_comparison.set_index('date')['price'].resample('ME').last()
                                spx_returns = spx_monthly.pct_change().dropna()
                                
                                # Align dates
                                common_dates = stock_returns.index.intersection(spx_returns.index)
                                
                                if len(common_dates) > 0:
                                    # Create comparison dataframe
                                    comparison_df = pd.DataFrame({
                                        'Date': common_dates,
                                        f'{ticker_upper} Return': stock_returns.loc[common_dates] * 100,
                                        'S&P 500 Return': spx_returns.loc[common_dates] * 100,
                                    })
                                    
                                    # Calculate outperformance
                                    comparison_df['Outperformance'] = comparison_df[f'{ticker_upper} Return'] - comparison_df['S&P 500 Return']
                                    
                                    # Format dates for display
                                    comparison_df['Date'] = comparison_df['Date'].dt.strftime('%Y-%m')
                                    
                                    # Sort by date (newest first)
                                    comparison_df = comparison_df.sort_values('Date', ascending=False)
                                    
                                    # Format the numerical columns for better display
                                    display_df = comparison_df.copy()
                                    display_df[f'{ticker_upper} Return'] = display_df[f'{ticker_upper} Return'].apply(lambda x: f"{x:.2f}%")
                                    display_df['S&P 500 Return'] = display_df['S&P 500 Return'].apply(lambda x: f"{x:.2f}%")
                                    display_df['Outperformance'] = display_df['Outperformance'].apply(lambda x: f"{x:.2f}%")
                                    
                                    # Display the table
                                    st.dataframe(
                                        display_df,
                                        use_container_width=True,
                                        height=400
                                    )
                                    
                                    # Summary statistics
                                    stock_mean = comparison_df[f'{ticker_upper} Return'].mean()
                                    spx_mean = comparison_df['S&P 500 Return'].mean()
                                    avg_outperformance = comparison_df['Outperformance'].mean()
                                    win_rate = (comparison_df['Outperformance'] > 0).mean() * 100
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Avg Monthly Return", f"{stock_mean:.2f}%")
                                    with col2:
                                        st.metric("S&P 500 Avg", f"{spx_mean:.2f}%")
                                    with col3:
                                        st.metric("Avg Outperformance", f"{avg_outperformance:.2f}%")
                                    with col4:
                                        st.metric("Win Rate", f"{win_rate:.1f}%")
                                    
                                    # Add scatterplot with beta analysis
                                    st.markdown("**📊 Beta Analysis Scatterplot**")
                                    
                                    # Calculate beta and correlation for the stock
                                    stock_returns_for_beta = stock_returns.loc[common_dates].values
                                    spx_returns_for_beta = spx_returns.loc[common_dates].values
                                    
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(spx_returns_for_beta, stock_returns_for_beta)
                                    
                                    # Create scatterplot
                                    fig_scatter = go.Figure()
                                    
                                    # Add scatter points
                                    fig_scatter.add_trace(go.Scatter(
                                        x=spx_returns_for_beta * 100,  # Convert to percentages
                                        y=stock_returns_for_beta * 100,  # Convert to percentages
                                        mode='markers',
                                        name='Monthly Returns',
                                        marker=dict(
                                            size=8,
                                            color='blue',
                                            opacity=0.7,
                                            line=dict(width=1, color='darkblue')
                                        ),
                                        text=[f"Date: {date}<br>SPX: {spx:.2f}%<br>{ticker_upper}: {stock:.2f}%" 
                                              for date, spx, stock in zip(common_dates.strftime('%Y-%m'), 
                                                                        spx_returns_for_beta * 100, 
                                                                        stock_returns_for_beta * 100)],
                                        hovertemplate='%{text}<extra></extra>'
                                    ))
                                    
                                    # Add beta trendline
                                    line_x = np.array([spx_returns_for_beta.min(), spx_returns_for_beta.max()]) * 100
                                    line_y = (slope * np.array([spx_returns_for_beta.min(), spx_returns_for_beta.max()]) + intercept) * 100
                                    
                                    fig_scatter.add_trace(go.Scatter(
                                        x=line_x,
                                        y=line_y,
                                        mode='lines',
                                        name=f'Beta Trendline (β={slope:.2f})',
                                        line=dict(color='red', width=3, dash='dash')
                                    ))
                                    
                                    # Update layout
                                    fig_scatter.update_layout(
                                        title=f'{ticker_upper} Beta Analysis: β = {slope:.2f}, R² = {r_value**2:.3f}',
                                        xaxis_title='S&P 500 Monthly Returns (%)',
                                        yaxis_title=f'{ticker_upper} Monthly Returns (%)',
                                        height=500,
                                        showlegend=True,
                                        hovermode='closest'
                                    )
                                    
                                    # Add reference lines at 0
                                    fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                                    fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
                                    
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                    
                                    # Display beta metrics
                                    beta_col1, beta_col2, beta_col3, beta_col4 = st.columns(4)
                                    with beta_col1:
                                        st.metric("Beta", f"{slope:.3f}")
                                    with beta_col2:
                                        st.metric("R-squared", f"{r_value**2:.3f}")
                                    with beta_col3:
                                        st.metric("Correlation", f"{r_value:.3f}")
                                    with beta_col4:
                                        st.metric("Data Points", len(common_dates))
                                    
                                    # Beta interpretation
                                    if slope > 1.2:
                                        st.error(f"🔴 High Beta: {ticker_upper} is {((slope-1)*100):+.0f}% more volatile than the market")
                                    elif slope > 0.8:
                                        st.info(f"🟡 Moderate Beta: {ticker_upper} volatility is {((slope-1)*100):+.0f}% vs market")
                                    else:
                                        st.success(f"🟢 Low Beta: {ticker_upper} is {((1-slope)*100):.0f}% less volatile than the market")
                                        
                                else:
                                    st.error("No overlapping dates found between stock and S&P 500 data")
                            else:
                                st.error(f"Failed to fetch historical data for {ticker_upper}")
                        except Exception as e:
                            # Log the error but provide a user-friendly message
                            error_msg = str(e)
                            if "background_gradient" in error_msg or "matplotlib" in error_msg:
                                st.error("Error calculating monthly returns: Styling feature not available in this environment.")
                            else:
                                st.error(f"Error calculating monthly returns: Unable to fetch or process data for {ticker_upper}")
                            st.info("💡 Try using the basic stock information section above instead.")
                else:
                    st.error(f"Could not fetch data for {ticker_upper}")

    with tab6:
        st.subheader("💰 Stock Valuation")
        
        # Input methods
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Manual ticker entry
            stock_input = st.text_input(
                "Enter stock tickers (comma-separated):",
                placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA",
                help="Enter multiple stock tickers separated by commas"
            )
        
        with col2:
            st.markdown("**Quick Select:**")
            quick_select = st.selectbox(
                "Choose group:",
                ["Custom", "FAANG", "Tech Giants", "Portfolio Stocks"],
                help="Select a predefined group of stocks"
            )
        
        # Handle quick select
        tickers_to_analyze = []
        
        if quick_select == "FAANG":
            tickers_to_analyze = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]
            st.info("Selected FAANG stocks")
        elif quick_select == "Tech Giants":
            tickers_to_analyze = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
            st.info("Selected Tech Giants")
        elif quick_select == "Portfolio Stocks":
            portfolio_positions = load_portfolio_positions()
            if portfolio_positions is not None:
                tickers_to_analyze = portfolio_positions['Symbol'].tolist()  # Include all portfolio stocks
                st.info(f"Selected all {len(tickers_to_analyze)} portfolio stocks: {', '.join(tickers_to_analyze)}")
            else:
                st.warning("No portfolio data available")
        else:
            if stock_input:
                tickers_to_analyze = [ticker.strip().upper() for ticker in stock_input.split(',')]
        
        # Analysis button
        if st.button("🔍 Analyze Stocks", disabled=len(tickers_to_analyze) == 0):
            if len(tickers_to_analyze) > 0:
                st.markdown(f"### Analysis Results for: {', '.join(tickers_to_analyze)}")
                
                # Fetch data for all stocks
                stock_data_list = []
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(tickers_to_analyze):
                    with st.spinner(f"Fetching data for {ticker}..."):
                        data = fetch_stock_data(ticker)
                        if data:
                            stock_data_list.append(data)
                    progress_bar.progress((i + 1) / len(tickers_to_analyze))
                
                progress_bar.empty()
                
                if stock_data_list:
                    # Create comparison DataFrame
                    comparison_data = []
                    
                    for data in stock_data_list:
                        comparison_data.append({
                            'Symbol': data['symbol'],
                            'Company': data['name'][:30] + "..." if len(data['name']) > 30 else data['name'],
                            'Price': format_currency(data['price']),
                            'Market Cap': format_currency(data['market_cap']),
                            'P/E': f"{data['pe_ratio']:.1f}" if data['pe_ratio'] else "N/A",
                            'P/B': f"{data['pb_ratio']:.1f}" if data['pb_ratio'] else "N/A",
                            'P/FCF': f"{data['p_fcf_ratio']:.1f}" if data['p_fcf_ratio'] else "N/A",
                            'ROE': format_percentage(data['roe']),
                            'Div Yield': format_dividend_yield(data['dividend_yield']),
                            'Beta': f"{data['beta']:.2f}" if data['beta'] else "N/A",
                            'FCF': format_currency(data['free_cash_flow']),
                            'FCF Method': data['fcf_method']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display results
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Insights
                    st.markdown("### 🎯 Key Insights")
                    
                    # Find interesting metrics
                    valid_pe_data = [d for d in stock_data_list if d['pe_ratio'] and d['pe_ratio'] > 0]
                    valid_roe_data = [d for d in stock_data_list if d['roe'] and d['roe'] > 0]
                    valid_div_data = [d for d in stock_data_list if d['dividend_yield'] and d['dividend_yield'] > 0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if valid_pe_data:
                            lowest_pe = min(valid_pe_data, key=lambda x: x['pe_ratio'])
                            highest_pe = max(valid_pe_data, key=lambda x: x['pe_ratio'])
                            st.metric("Lowest P/E", f"{lowest_pe['symbol']}: {lowest_pe['pe_ratio']:.1f}")
                            st.metric("Highest P/E", f"{highest_pe['symbol']}: {highest_pe['pe_ratio']:.1f}")
                    
                    with col2:
                        if valid_roe_data:
                            highest_roe = max(valid_roe_data, key=lambda x: x['roe'])
                            st.metric("Highest ROE", f"{highest_roe['symbol']}: {highest_roe['roe']*100:.1f}%")
                    
                    with col3:
                        if valid_div_data:
                            highest_div = max(valid_div_data, key=lambda x: x['dividend_yield'])
                            st.metric("Highest Dividend", f"{highest_div['symbol']}: {highest_div['dividend_yield']:.2f}%")
                    
                    # Export functionality
                    csv_data = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=f"stock_valuation_{'-'.join(tickers_to_analyze)}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("Could not fetch data for any of the selected stocks")
            else:
                st.warning("Please enter at least one stock ticker")
        
        # DCF Calculator section
        st.markdown("---")
        st.markdown("### 🧮 DCF Calculator")
        
        dcf_ticker = st.text_input(
            "Enter ticker for DCF analysis:",
            placeholder="e.g., AAPL",
            help="Calculate intrinsic value using Discounted Cash Flow method"
        )
        
        if dcf_ticker:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**DCF Assumptions**")
                growth_rate = st.number_input("High Growth Rate (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5) / 100
                growth_years = st.number_input("Growth Years", min_value=1, max_value=15, value=5, step=1)
            
            with col2:
                terminal_growth = st.number_input("Terminal Growth Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1) / 100
                wacc = st.number_input("WACC/Discount Rate (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.5) / 100
            
            if st.button(f"📊 Calculate DCF for {dcf_ticker.upper()}"):
                with st.spinner("Calculating DCF valuation..."):
                    dcf_result, status = calculate_dcf_valuation(
                        dcf_ticker.upper(), growth_rate, growth_years, terminal_growth, wacc
                    )
                
                if dcf_result and status == "success":
                    st.success("✅ DCF Calculation Complete")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Price", format_currency(dcf_result['current_price']))
                        st.metric("Intrinsic Value", format_currency(dcf_result['intrinsic_value']))
                        
                    with col2:
                        upside = dcf_result['upside_downside']
                        if upside:
                            upside_pct = upside * 100
                            if upside > 0.2:
                                st.success(f"🚀 Undervalued: {upside_pct:.1f}% upside")
                            elif upside > 0:
                                st.info(f"📈 Fair Value: {upside_pct:.1f}% upside")
                            elif upside > -0.2:
                                st.warning(f"⚖️ Fair Value: {abs(upside_pct):.1f}% downside")
                            else:
                                st.error(f"📉 Overvalued: {abs(upside_pct):.1f}% downside")
                    
                    # Show assumptions used
                    with st.expander("📋 View DCF Assumptions & Details"):
                        assumptions = dcf_result['assumptions']
                        st.write(f"**Growth Rate**: {assumptions['growth_rate']*100:.1f}%")
                        st.write(f"**Growth Years**: {assumptions['growth_years']}")
                        st.write(f"**Terminal Growth**: {assumptions['terminal_growth']*100:.1f}%")
                        st.write(f"**WACC**: {assumptions['wacc']*100:.1f}%")
                        st.write(f"**Current FCF**: {format_currency(assumptions['current_fcf'])}")
                        st.write(f"**FCF Data Source**: {assumptions['fcf_method']}")
                        st.write(f"**Enterprise Value**: {format_currency(dcf_result['enterprise_value'])}")
                        st.write(f"**Equity Value**: {format_currency(dcf_result['equity_value'])}")
                
                else:
                    st.error(f"DCF calculation failed: {status}")
        
        if not stock_input and quick_select == "Custom":
            st.info("💡 **Getting Started**: Enter stock tickers above or use quick select to begin analysis")

    with tab7:
        st.subheader("⚡ Markowitz Efficient Frontier Analysis")
        st.markdown("**Build optimal portfolios using Modern Portfolio Theory**")
        
        # Stock selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Select Stocks for Analysis** (Up to 50 equities)")
            
            # Pre-defined stock lists for quick selection
            preset_options = {
                "Custom Selection": [],
                "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"],
                "Dividend Aristocrats": ["JNJ", "PG", "KO", "PEP", "MCD", "WMT", "HD", "MMM"],
                "S&P 500 Core": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "UNH"],
                "Your Portfolio Holdings": []  # Will be populated dynamically
            }
            
            # Load portfolio positions to populate "Your Portfolio Holdings"
            portfolio_positions = load_portfolio_positions()
            if portfolio_positions is not None and len(portfolio_positions) > 0:
                # Filter out index funds and ETFs for individual stock analysis
                index_fund_patterns = ['SPY', 'VOO', 'IVV', 'VTI', 'QQQ', 'DIA', 'EFA', 'EEM', 'VEA', 'VWO', 'BND', 'AGG', 'VXUS', 'VTEB', 'SPAXX', 'VTIAX', 'VTMGX', 'SPDR', 'ETF', 'VTV', 'IHI']
                individual_stocks = []
                
                for _, position in portfolio_positions.iterrows():
                    symbol = position['Symbol']
                    if pd.isna(symbol) or symbol == 'SPAXX**':  # Skip cash positions and NaN
                        continue
                    
                    # More comprehensive filtering to exclude index funds/ETFs
                    symbol_upper = str(symbol).upper()
                    is_index_fund = any(pattern in symbol_upper for pattern in index_fund_patterns)
                    is_etf = symbol_upper.endswith('X') or len(symbol) > 4  # Common ETF patterns
                    
                    if not is_index_fund and not is_etf and len(symbol) <= 5:
                        individual_stocks.append(symbol)
                
                if individual_stocks:
                    preset_options["Your Portfolio Holdings"] = individual_stocks  # Include all individual stocks
                    st.info(f"📊 Found {len(individual_stocks)} individual stocks in your portfolio (filtered {len(portfolio_positions) - len(individual_stocks)} index funds/ETFs)")
                else:
                    st.info("📊 No individual stocks found in portfolio (only index funds/ETFs detected)")
            else:
                st.info("📊 No portfolio data available - using preset options only")
            
            preset_choice = st.selectbox("Quick Select Portfolio:", list(preset_options.keys()))
            
            if preset_choice != "Custom Selection":
                default_tickers = ", ".join(preset_options[preset_choice])
            else:
                default_tickers = ""
            
            efficient_frontier_tickers = st.text_area(
                "Enter stock tickers (comma-separated):",
                value=default_tickers,
                placeholder="e.g., AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JNJ",
                help="Maximum 50 stocks recommended for optimal performance"
            )
        
        with col2:
            st.markdown("**Analysis Parameters**")
            
            ef_risk_free_rate = st.number_input(
                "Risk-Free Rate (Annual %)",
                min_value=0.0,
                max_value=10.0,
                value=3.044,
                step=0.01,
                format="%.3f",
                key="ef_risk_free"
            ) / 100
            
            # Date range selection for historical data
            st.markdown("**Historical Data Period**")
            
            # Create two columns for start and end dates
            date_col1, date_col2 = st.columns(2)
            
            with date_col1:
                st.markdown("**Start Date**")
                start_year = st.selectbox(
                    "Year",
                    options=list(range(1970, 2026)),
                    index=50,  # Default to 2020 (1970 + 50 = 2020)
                    key="ef_start_year"
                )
                
                start_month = st.selectbox(
                    "Month",
                    options=list(range(1, 13)),
                    index=0,  # Default to January
                    format_func=lambda x: pd.to_datetime(f"2023-{x:02d}-01").strftime("%B"),
                    key="ef_start_month"
                )
            
            with date_col2:
                st.markdown("**End Date**")
                end_year = st.selectbox(
                    "Year",
                    options=list(range(1970, 2026)),
                    index=55,  # Default to 2025 (1970 + 55 = 2025)
                    key="ef_end_year"
                )
                
                end_month = st.selectbox(
                    "Month",
                    options=list(range(1, 13)),
                    index=11,  # Default to December
                    format_func=lambda x: pd.to_datetime(f"2023-{x:02d}-01").strftime("%B"),
                    key="ef_end_month"
                )
            
            # Create start and end dates
            start_date_ef = pd.to_datetime(f"{start_year}-{start_month:02d}-01")
            end_date_ef = pd.to_datetime(f"{end_year}-{end_month:02d}-01") + pd.offsets.MonthEnd()
            
            # Validate date range
            if start_date_ef >= end_date_ef:
                st.error("❌ Start date must be before end date")
                date_range_valid = False
            else:
                months_diff = (end_date_ef.year - start_date_ef.year) * 12 + (end_date_ef.month - start_date_ef.month)
                if months_diff < 24:
                    st.warning(f"⚠️ Only {months_diff} months selected. Minimum 24 months recommended for reliable optimization.")
                    date_range_valid = True  # Still allow analysis but with warning
                elif months_diff < 12:
                    st.error(f"❌ Only {months_diff} months selected. Minimum 12 months required.")
                    date_range_valid = False
                else:
                    st.success(f"✅ {months_diff} months of data selected ({start_date_ef.strftime('%b %Y')} - {end_date_ef.strftime('%b %Y')})")
                    date_range_valid = True
            
            num_portfolios = st.slider(
                "Frontier Points",
                min_value=20,
                max_value=100,
                value=50,
                step=10,
                help="Number of portfolios on efficient frontier"
            )
        
        if efficient_frontier_tickers and date_range_valid:
            # Parse and validate tickers
            tickers_list = [ticker.strip().upper() for ticker in efficient_frontier_tickers.split(",") if ticker.strip()]
            
            if len(tickers_list) > 50:
                st.warning("⚠️ Maximum 50 stocks recommended. Using first 50 tickers.")
                tickers_list = tickers_list[:50]
            
            if len(tickers_list) < 2:
                st.error("❌ Please enter at least 2 stock tickers for portfolio optimization")
            else:
                st.success(f"✅ Analyzing {len(tickers_list)} stocks: {', '.join(tickers_list)}")
                st.info(f"📅 Data period: {start_date_ef.strftime('%B %Y')} to {end_date_ef.strftime('%B %Y')}")
                
                # Initialize session state for efficient frontier results
                if 'ef_results' not in st.session_state:
                    st.session_state.ef_results = None
                    st.session_state.ef_tickers = None
                    st.session_state.ef_beta_data = None
                    st.session_state.ef_annual_returns = None
                    st.session_state.ef_cov_matrix = None
                    st.session_state.ef_risk_free_rate = None
                
                if st.button("🚀 Generate Efficient Frontier", type="primary"):
                    with st.spinner("Fetching historical data and optimizing portfolios..."):
                        
                        # Fetch historical data for the specified date range
                        price_data, status = get_historical_data_custom_range(tickers_list, start_date_ef, end_date_ef)
                        
                        if status != "success":
                            st.error(f"❌ Failed to fetch data: {status}")
                        else:
                            # Calculate returns and statistics
                            monthly_returns, annual_returns, cov_matrix, calc_status = calculate_returns_and_stats(price_data)
                            
                            if calc_status != "success":
                                st.error(f"❌ Failed to calculate statistics: {calc_status}")
                            else:
                                # Calculate stock betas for CAPM analysis
                                beta_data = None
                                beta_status = "failed"
                                
                                with st.spinner("Calculating stock betas for CAPM analysis..."):
                                    try:
                                        beta_data, beta_status = calculate_stock_betas(tickers_list, start_date_ef, end_date_ef)
                                        if beta_status == "success" and beta_data:
                                            st.success(f"✅ Beta analysis completed for {len(beta_data)} stocks")
                                        else:
                                            st.warning(f"⚠️ Beta calculation issues: {beta_status}")
                                    except Exception as e:
                                        st.error(f"❌ Beta calculation failed: {str(e)}")
                                        beta_data = None
                                        beta_status = f"error: {str(e)}"
                                
                                # Optimize portfolios
                                optimization_results, opt_status = optimize_portfolio(
                                    annual_returns, cov_matrix, ef_risk_free_rate, num_portfolios
                                )
                                
                                if opt_status != "success":
                                    st.error(f"❌ Optimization failed: {opt_status}")
                                else:
                                    st.success("✅ Efficient Frontier Generated Successfully!")
                                    
                                    # Store results in session state for Monte Carlo simulation
                                    st.session_state.ef_results = optimization_results
                                    st.session_state.ef_tickers = tickers_list.copy()
                                    st.session_state.ef_beta_data = beta_data
                                    st.session_state.ef_annual_returns = annual_returns
                                    st.session_state.ef_cov_matrix = cov_matrix
                                    st.session_state.ef_risk_free_rate = ef_risk_free_rate
                                    st.session_state.ef_monthly_returns = monthly_returns  # Store monthly returns for export
                
                # Display results if available (either just generated or from session state)
                if st.session_state.ef_results is not None:
                    optimization_results = st.session_state.ef_results
                    tickers_list = st.session_state.ef_tickers if st.session_state.ef_tickers is not None else []
                    beta_data = st.session_state.ef_beta_data if st.session_state.ef_beta_data is not None else {}
                    annual_returns = st.session_state.ef_annual_returns
                    cov_matrix = st.session_state.ef_cov_matrix
                    ef_risk_free_rate = st.session_state.ef_risk_free_rate
                    
                    # Store covariance matrix and beta data for analysis
                    optimization_results['cov_matrix'] = cov_matrix
                    optimization_results['beta_data'] = beta_data
                    
                    # Display summary statistics including beta analysis
                    opt_stats = optimization_results['statistics']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Portfolios Generated", opt_stats['num_portfolios_generated'])
                    with col2:
                        min_ret, max_ret = opt_stats['target_return_range']
                        st.metric("Return Range", f"{min_ret*100:.1f}% - {max_ret*100:.1f}%")
                    with col3:
                        # Use a default value for monthly data points if not available
                        st.metric("Stocks Analyzed", len(tickers_list))
                    with col4:
                        if beta_data and len(beta_data) > 0:
                            try:
                                # More robust beta calculation with multiple safety checks
                                valid_betas = []
                                for data in beta_data.values():
                                    if (data and 
                                        isinstance(data, dict) and 
                                        'beta' in data and 
                                        data['beta'] is not None and
                                        isinstance(data['beta'], (int, float)) and
                                        not pd.isna(data['beta'])):
                                        valid_betas.append(data['beta'])
                                
                                if valid_betas:
                                    avg_beta = pd.Series(valid_betas).mean()
                                    st.metric("Average Stock Beta", f"{avg_beta:.2f}")
                                else:
                                    st.metric("Beta Analysis", "N/A (No valid data)")
                            except (KeyError, TypeError, ValueError, AttributeError) as e:
                                st.metric("Beta Analysis", "N/A (Error)")
                        else:
                            st.metric("Beta Analysis", "N/A")
                    
                    # Create and display plot
                    fig = create_efficient_frontier_plot(
                        optimization_results, annual_returns, ef_risk_free_rate, tickers_list, None
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display portfolio summaries
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🟢 Minimum Volatility Portfolio")
                        min_vol = optimization_results['min_volatility_portfolio']
                        
                        st.metric("Expected Return", f"{min_vol['return']*100:.2f}%")
                        st.metric("Volatility", f"{min_vol['volatility']*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{min_vol['sharpe_ratio']:.3f}")
                        
                        # Calculate and display portfolio beta
                        if beta_data:
                            min_vol_beta = calculate_portfolio_beta(min_vol['weights'], beta_data, tickers_list)
                            st.metric("Portfolio Beta", f"{min_vol_beta:.2f}")
                        
                        # Show top holdings
                        weights_df = pd.DataFrame({
                            'Stock': tickers_list,
                            'Weight': min_vol['weights']
                        }).sort_values('Weight', ascending=False)
                        
                        # Filter out very small weights
                        significant_weights = weights_df[weights_df['Weight'] > 0.01]
                        
                        st.markdown("**Top Holdings (>1%)**")
                        for _, row in significant_weights.head(8).iterrows():
                            st.write(f"• {row['Stock']}: {row['Weight']*100:.1f}%")
                    
                    with col2:
                        st.markdown("### 🟡 Maximum Sharpe Ratio Portfolio")
                        max_sharpe = optimization_results['max_sharpe_portfolio']
                        
                        st.metric("Expected Return", f"{max_sharpe['return']*100:.2f}%")
                        st.metric("Volatility", f"{max_sharpe['volatility']*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe_ratio']:.3f}")
                        
                        # Calculate and display portfolio beta
                        if beta_data:
                            max_sharpe_beta = calculate_portfolio_beta(max_sharpe['weights'], beta_data, tickers_list)
                            st.metric("Portfolio Beta", f"{max_sharpe_beta:.2f}")
                        
                        # Show top holdings
                        weights_df = pd.DataFrame({
                            'Stock': tickers_list,
                            'Weight': max_sharpe['weights']
                        }).sort_values('Weight', ascending=False)
                        
                        # Filter out very small weights
                        significant_weights = weights_df[weights_df['Weight'] > 0.01]
                        
                        st.markdown("**Top Holdings (>1%)**")
                        for _, row in significant_weights.head(8).iterrows():
                            st.write(f"• {row['Stock']}: {row['Weight']*100:.1f}%")
                    
                    # Detailed portfolio weights table
                    with st.expander("📊 View Complete Portfolio Weights"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Minimum Volatility Portfolio**")
                            min_vol_weights = pd.DataFrame({
                'Stock': tickers_list,
                'Weight (%)': [w*100 for w in min_vol['weights']]
                            }).sort_values('Weight (%)', ascending=False)
                            st.dataframe(min_vol_weights, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Maximum Sharpe Ratio Portfolio**")
                            max_sharpe_weights = pd.DataFrame({
                'Stock': tickers_list,
                'Weight (%)': [w*100 for w in max_sharpe['weights']]
                            }).sort_values('Weight (%)', ascending=False)
                            st.dataframe(max_sharpe_weights, use_container_width=True)
                    
                    # Export functionality
                    with st.expander("📥 Export Results"):
                        
                        # Prepare data for export
                        efficient_frontier_data = []
                        for i, portfolio in enumerate(optimization_results['efficient_frontier']):
                            row = {
                'Portfolio': f'EF_{i+1:02d}',
                'Expected_Return_%': portfolio['return'] * 100,
                'Volatility_%': portfolio['volatility'] * 100,
                'Sharpe_Ratio': portfolio['sharpe_ratio']
                            }
                            # Add individual weights
                            for j, ticker in enumerate(tickers_list):
                                row[f'{ticker}_Weight_%'] = portfolio['weights'][j] * 100
                            efficient_frontier_data.append(row)
                        
                        ef_df = pd.DataFrame(efficient_frontier_data)
                        
                        # Download buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            csv_data = ef_df.to_csv(index=False)
                            st.download_button(
                label="📊 Download Efficient Frontier Data",
                data=csv_data,
                file_name=f"efficient_frontier_{'-'.join(tickers_list[:5])}.csv",
                mime="text/csv"
                            )
                        
                        with col2:
                            # Add monthly returns download option
                            # Access monthly returns from session state calculation
                            if hasattr(st.session_state, 'ef_monthly_returns') and st.session_state.ef_monthly_returns is not None:
                                monthly_returns_data = st.session_state.ef_monthly_returns
                                
                                # Create monthly returns DataFrame for export
                                monthly_returns_export = monthly_returns_data.copy()
                                monthly_returns_export.index = monthly_returns_export.index.strftime('%Y-%m')
                                monthly_returns_export = monthly_returns_export * 100  # Convert to percentages
                                monthly_returns_export = monthly_returns_export.round(3)  # Round to 3 decimal places
                                
                                # Add a date column
                                monthly_returns_export = monthly_returns_export.reset_index()
                                monthly_returns_export.rename(columns={'index': 'Date'}, inplace=True)
                                
                                monthly_returns_csv = monthly_returns_export.to_csv(index=False)
                                
                                st.download_button(
                                    label="📈 Download Monthly Returns (%)",
                                    data=monthly_returns_csv,
                                    file_name=f"monthly_returns_{'-'.join(tickers_list[:5])}.csv",
                                    mime="text/csv",
                                    help="Monthly percentage returns used to calculate efficient frontier. Use formula (1+average_monthly_return)^12-1 to verify annualized returns."
                                )
                            else:
                                st.info("💡 Monthly returns data not available - run analysis first")
                        
                        with col3:
                            # Calculate portfolio betas for optimal portfolios
                            min_vol_beta = calculate_portfolio_beta(min_vol['weights'], beta_data, tickers_list) if beta_data else 'N/A'
                            max_sharpe_beta = calculate_portfolio_beta(max_sharpe['weights'], beta_data, tickers_list) if beta_data else 'N/A'
                            
                            report_data = {
                'Analysis_Date': [pd.Timestamp.now().strftime('%Y-%m-%d')],
                'Risk_Free_Rate_%': [ef_risk_free_rate * 100],
                'Number_of_Stocks': [len(tickers_list)],
                'Stocks_Analyzed': [', '.join(tickers_list)],
                'Min_Vol_Return_%': [min_vol['return'] * 100],
                'Min_Vol_Risk_%': [min_vol['volatility'] * 100],
                'Min_Vol_Sharpe': [min_vol['sharpe_ratio']],
                'Min_Vol_Beta': [min_vol_beta if min_vol_beta != 'N/A' else 'N/A'],
                'Max_Sharpe_Return_%': [max_sharpe['return'] * 100],
                'Max_Sharpe_Risk_%': [max_sharpe['volatility'] * 100],
                'Max_Sharpe_Sharpe': [max_sharpe['sharpe_ratio']],
                'Max_Sharpe_Beta': [max_sharpe_beta if max_sharpe_beta != 'N/A' else 'N/A'],
                'Beta_Analysis_Available': ['Yes' if beta_data else 'No']
                            }
                            
                            report_df = pd.DataFrame(report_data)
                            report_csv = report_df.to_csv(index=False)
                            
                            st.download_button(
                label="📄 Download Analysis Summary",
                data=report_csv,
                file_name=f"portfolio_optimization_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
                            )
                    
                    # Risk-return and beta statistics
                    with st.expander("📈 Individual Stock Statistics & CAPM Analysis"):
                        stats_data = []
                        individual_returns = optimization_results['statistics']['individual_asset_returns']
                        individual_volatilities = optimization_results['statistics']['individual_asset_volatilities']
                        
                        for i, ticker in enumerate(tickers_list):
                            annual_return = individual_returns[i]
                            annual_volatility = individual_volatilities[i]
                            sharpe_ratio = (annual_return - ef_risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
                            
                            # Add beta information if available
                            beta_info = beta_data.get(ticker, {}) if beta_data else {}
                            beta = beta_info.get('beta', 'N/A')
                            r_squared = beta_info.get('r_squared', 'N/A')
                            
                            # Calculate CAPM expected return if beta is available
                            capm_return = 'N/A'
                            if isinstance(beta, (int, float)) and not pd.isna(beta):
                                # CAPM: E(R) = Rf + β(Rm - Rf)
                                # Assume market return of 10% for calculation
                                market_return = 0.10
                                capm_expected = ef_risk_free_rate + beta * (market_return - ef_risk_free_rate)
                                capm_return = f"{capm_expected * 100:.1f}%"
                            
                            stats_data.append({
                                'Stock': ticker,
                                'Expected Return (%)': f"{annual_return * 100:.1f}%",
                                'Volatility (%)': f"{annual_volatility * 100:.1f}%",
                                'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                                'Beta': f"{beta:.2f}" if isinstance(beta, (int, float)) and not pd.isna(beta) else 'N/A',
                                'R-Squared': f"{r_squared:.3f}" if isinstance(r_squared, (int, float)) and not pd.isna(r_squared) else 'N/A',
                                'CAPM Expected Return': capm_return
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        
                        # Sort by Sharpe Ratio (convert back to numeric for sorting)
                        stats_df['Sharpe_Numeric'] = [float(x) if x != 'N/A' else -999 for x in stats_df['Sharpe Ratio']]
                        stats_df = stats_df.sort_values('Sharpe_Numeric', ascending=False).drop('Sharpe_Numeric', axis=1)
                        
                        st.dataframe(stats_df, use_container_width=True)
                        
                        if beta_data:
                            st.markdown("**📊 Beta Analysis Summary:**")
                            betas = [data['beta'] for data in beta_data.values()]
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Lowest Beta", f"{min(betas):.2f}")
                            with col2:
                                st.metric("Highest Beta", f"{max(betas):.2f}")
                            with col3:
                                st.metric("Average Beta", f"{pd.Series(betas).mean():.2f}")
                            with col4:
                                high_beta_count = sum(1 for b in betas if b > 1.0)
                                st.metric("High Beta Stocks", f"{high_beta_count}/{len(betas)}")
                        else:
                            st.info("💡 Beta calculations not available for this analysis period")
                
                # Optimal Beta Strategy Section (moved before Monte Carlo)
                st.markdown("---")
                # Check if we have beta data (either from current session or session state)
                if hasattr(st.session_state, 'ef_beta_data') and st.session_state.ef_beta_data:
                    current_beta_data = st.session_state.ef_beta_data
                elif 'beta_data' in locals() and beta_data:
                    current_beta_data = beta_data
                else:
                    current_beta_data = None
                
                if current_beta_data:
                    st.subheader("🎯 Optimal Beta Strategy")
                    st.markdown("Find the best risk-adjusted portfolio using beta-based analysis:")
                    
                    if st.button("🎯 Find Optimal Beta Strategy", type="primary"):
                        with st.spinner("Finding optimal beta strategy..."):
                            success = display_optimal_beta_strategy(
                                optimization_results, current_beta_data, annual_returns, 
                                cov_matrix, ef_risk_free_rate, tickers_list
                            )
                            
                            if success:
                                st.success("✅ Optimal beta strategy found!")
                            else:
                                st.error("❌ Beta strategy analysis failed")
                else:
                    st.subheader("🎯 Optimal Beta Strategy")
                    st.info("💡 Beta strategy analysis requires beta data - run the efficient frontier analysis first")
                
                # Monte Carlo Simulation Section
                st.markdown("---")
                st.subheader("🎯 Monte Carlo Portfolio Comparison")
                st.markdown("**Compare Markowitz Maximum Sharpe vs Beta Strategy head-to-head**")
                
                # Monte Carlo controls
                mc_col1, mc_col2, mc_col3 = st.columns(3)
                
                with mc_col1:
                    initial_investment = st.number_input(
                        "Initial Investment ($)",
                        min_value=1000,
                        max_value=10000000,
                        value=100000,
                        step=10000,
                        format="%d"
                    )
                
                with mc_col2:
                    time_horizon = st.slider(
                        "Time Horizon (Years)",
                        min_value=1,
                        max_value=30,
                        value=10,
                        help="Investment period for simulation"
                    )
                
                with mc_col3:
                    num_simulations = st.selectbox(
                        "Simulations",
                        [1000, 5000, 10000, 25000],
                        index=2,
                        help="More simulations = more accuracy but slower"
                    )
                
                # Check if beta strategy is available
                beta_strategy_available = (hasattr(st.session_state, 'best_beta_strategy') and 
                                         st.session_state.best_beta_strategy)
                
                if beta_strategy_available:
                    st.info(f"✅ Ready to compare: **Markowitz Max Sharpe** vs **{st.session_state.best_beta_strategy['name']}**")
                else:
                    st.warning("⚠️ Beta strategy not available. Run 'Find Optimal Beta Strategy' first to enable comparison.")
                
                # Run Monte Carlo comparison
                if st.button("🎲 Run Portfolio Comparison", type="primary", disabled=not beta_strategy_available):
                    # Ensure we have the required data from session state
                    if (st.session_state.ef_results is None or 
                        st.session_state.ef_annual_returns is None or 
                        st.session_state.ef_cov_matrix is None):
                        st.error("❌ Please run 'Generate Efficient Frontier' first to get the required portfolio data.")
                    else:
                        # Get data from session state
                        optimization_results = st.session_state.ef_results
                        annual_returns = st.session_state.ef_annual_returns
                        cov_matrix = st.session_state.ef_cov_matrix
                        max_sharpe = optimization_results['max_sharpe_portfolio']
                        
                        with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations for both strategies..."):
                            
                            # Get both portfolios
                            markowitz_portfolio = max_sharpe
                            beta_strategy = st.session_state.best_beta_strategy
                        
                        # Run simulations for both strategies
                        st.write("🔄 Simulating Markowitz Maximum Sharpe portfolio...")
                        markowitz_results, markowitz_status = run_monte_carlo_simulation(
                            markowitz_portfolio['weights'],
                            annual_returns,
                            cov_matrix,
                            initial_investment=initial_investment,
                            time_horizon=time_horizon,
                            num_simulations=num_simulations
                        )
                        
                        st.write("🔄 Simulating Beta Strategy portfolio...")
                        beta_results, beta_status = run_monte_carlo_simulation(
                            beta_strategy['weights'],
                            annual_returns,
                            cov_matrix,
                            initial_investment=initial_investment,
                            time_horizon=time_horizon,
                            num_simulations=num_simulations
                        )
                        
                        if markowitz_status == "success" and beta_status == "success":
                            st.success("✅ Portfolio comparison completed!")
                            
                            # Display comparative results in tabs
                            mc_tab1, mc_tab2, mc_tab3, mc_tab4 = st.tabs([
                                "📊 Side-by-Side Comparison", 
                                "📈 Simulation Paths", 
                                "📊 Distribution Comparison",
                                "📋 Detailed Statistics"
                            ])
                            
                            with mc_tab1:
                                st.markdown("### 🥊 **Head-to-Head Performance Comparison**")
                                
                                # Key metrics comparison
                                markowitz_stats = markowitz_results['statistics']
                                beta_stats = beta_results['statistics']
                                
                                # Comparison table
                                comparison_data = {
                                    'Metric': [
                                        'Expected Final Value',
                                        'Median Final Value',
                                        'Mean Annual Return',
                                        'Annual Volatility', 
                                        'Sharpe Ratio',
                                        'Best Case (95th %ile)',
                                        'Worst Case (5th %ile)',
                                        'Probability of Loss',
                                        'Value at Risk (5%)',
                                        'Mean CAGR'
                                    ],
                                    'Markowitz Max Sharpe': [
                                        f"${markowitz_stats['mean_final_value']:,.0f}",
                                        f"${markowitz_stats['median_final_value']:,.0f}",
                                        f"{markowitz_stats['portfolio_expected_return']*100:.2f}%",
                                        f"{markowitz_stats['portfolio_volatility']*100:.2f}%",
                                        f"{(markowitz_stats['portfolio_expected_return'] - ef_risk_free_rate) / markowitz_stats['portfolio_volatility']:.3f}",
                                        f"${markowitz_stats['percentiles']['95th']:,.0f}",
                                        f"${markowitz_stats['percentiles']['5th']:,.0f}",
                                        f"{markowitz_stats['prob_loss']*100:.1f}%",
                                        f"${markowitz_stats['percentiles']['5th']:,.0f}",
                                        f"{markowitz_stats['mean_cagr']*100:.2f}%"
                                    ],
                                    f"{beta_strategy['name']}": [
                                        f"${beta_stats['mean_final_value']:,.0f}",
                                        f"${beta_stats['median_final_value']:,.0f}",
                                        f"{beta_stats['portfolio_expected_return']*100:.2f}%",
                                        f"{beta_stats['portfolio_volatility']*100:.2f}%",
                                        f"{(beta_stats['portfolio_expected_return'] - ef_risk_free_rate) / beta_stats['portfolio_volatility']:.3f}",
                                        f"${beta_stats['percentiles']['95th']:,.0f}",
                                        f"${beta_stats['percentiles']['5th']:,.0f}",
                                        f"{beta_stats['prob_loss']*100:.1f}%",
                                        f"${beta_stats['percentiles']['5th']:,.0f}",
                                        f"{beta_stats['mean_cagr']*100:.2f}%"
                                    ]
                                }
                                
                                # Add winner indicators
                                comparison_data['Winner'] = []
                                
                                # Determine winners (higher is better except for volatility and prob of loss)
                                metrics_higher_better = [0, 1, 2, 4, 5, 9]  # indices where higher is better
                                metrics_lower_better = [3, 7]  # volatility, prob of loss
                                
                                for i in range(len(comparison_data['Metric'])):
                                    mark_val = markowitz_stats['mean_final_value'] if i == 0 else (
                                        markowitz_stats['median_final_value'] if i == 1 else (
                                        markowitz_stats['portfolio_expected_return'] if i == 2 else (
                                        markowitz_stats['portfolio_volatility'] if i == 3 else (
                                        (markowitz_stats['portfolio_expected_return'] - ef_risk_free_rate) / markowitz_stats['portfolio_volatility'] if i == 4 else (
                                        markowitz_stats['percentiles']['95th'] if i == 5 else (
                                        markowitz_stats['percentiles']['5th'] if i == 6 else (
                                        markowitz_stats['prob_loss'] if i == 7 else (
                                        markowitz_stats['percentiles']['5th'] if i == 8 else 
                                        markowitz_stats['mean_cagr']))))))))
                                    
                                    beta_val = beta_stats['mean_final_value'] if i == 0 else (
                                        beta_stats['median_final_value'] if i == 1 else (
                                        beta_stats['portfolio_expected_return'] if i == 2 else (
                                        beta_stats['portfolio_volatility'] if i == 3 else (
                                        (beta_stats['portfolio_expected_return'] - ef_risk_free_rate) / beta_stats['portfolio_volatility'] if i == 4 else (
                                        beta_stats['percentiles']['95th'] if i == 5 else (
                                        beta_stats['percentiles']['5th'] if i == 6 else (
                                        beta_stats['prob_loss'] if i == 7 else (
                                        beta_stats['percentiles']['5th'] if i == 8 else 
                                        beta_stats['mean_cagr']))))))))
                                    
                                    if i in metrics_higher_better:
                                        winner = "Markowitz" if mark_val > beta_val else "Beta Strategy"
                                    elif i in metrics_lower_better:
                                        winner = "Markowitz" if mark_val < beta_val else "Beta Strategy"
                                    else:
                                        winner = "Markowitz" if mark_val > beta_val else "Beta Strategy"
                                    
                                    comparison_data['Winner'].append(f"🏆 {winner}" if winner else "📍 Tie")
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                                
                                # Summary winner
                                markowitz_wins = sum(1 for w in comparison_data['Winner'] if 'Markowitz' in w)
                                beta_wins = sum(1 for w in comparison_data['Winner'] if 'Beta Strategy' in w)
                                
                                if markowitz_wins > beta_wins:
                                    st.success(f"🏆 **Overall Winner: Markowitz Maximum Sharpe** ({markowitz_wins} wins vs {beta_wins})")
                                elif beta_wins > markowitz_wins:
                                    st.success(f"🏆 **Overall Winner: {beta_strategy['name']}** ({beta_wins} wins vs {markowitz_wins})")
                                else:
                                    st.info(f"🤝 **Tie Game!** Both strategies won {markowitz_wins} metrics each")
                            
                            with mc_tab2:
                                st.markdown("### 📈 **Simulation Path Comparison**")
                                
                                # Create combined simulation plot
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=[
                                        'Markowitz Maximum Sharpe Portfolio Paths',
                                        f'{beta_strategy["name"]} Portfolio Paths'
                                    ],
                                    vertical_spacing=0.12
                                )
                                
                                # Sample paths for visualization (show 50 random paths)
                                sample_indices = np.random.choice(num_simulations, min(50, num_simulations), replace=False)
                                
                                # Markowitz paths
                                for idx in sample_indices:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=list(range(time_horizon + 1)),
                                            y=markowitz_results['simulation_paths'][idx],
                                            mode='lines',
                                            line=dict(color='blue', width=1),
                                            opacity=0.3,
                                            showlegend=False,
                                            hovertemplate='Year %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                                        ),
                                        row=1, col=1
                                    )
                                
                                # Beta strategy paths
                                for idx in sample_indices:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=list(range(time_horizon + 1)),
                                            y=beta_results['simulation_paths'][idx],
                                            mode='lines',
                                            line=dict(color='red', width=1),
                                            opacity=0.3,
                                            showlegend=False,
                                            hovertemplate='Year %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                                        ),
                                        row=2, col=1
                                    )
                                
                                # Add median paths
                                markowitz_median_path = np.median(markowitz_results['simulation_paths'], axis=0)
                                beta_median_path = np.median(beta_results['simulation_paths'], axis=0)
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(range(time_horizon + 1)),
                                        y=markowitz_median_path,
                                        mode='lines',
                                        line=dict(color='darkblue', width=3),
                                        name='Markowitz Median Path'
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(range(time_horizon + 1)),
                                        y=beta_median_path,
                                        mode='lines',
                                        line=dict(color='darkred', width=3),
                                        name='Beta Strategy Median Path'
                                    ),
                                    row=2, col=1
                                )
                                
                                fig.update_layout(
                                    height=800,
                                    title_text="Portfolio Simulation Paths Comparison",
                                    showlegend=True
                                )
                                
                                fig.update_xaxes(title_text="Years")
                                fig.update_yaxes(title_text="Portfolio Value ($)")
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with mc_tab3:
                                st.markdown("### 📊 **Final Value Distribution Comparison**")
                                
                                # Create comparative distribution plot
                                import plotly.graph_objects as go  # Explicit import to avoid scope issues
                                fig = go.Figure()
                                
                                # Add histograms
                                fig.add_trace(go.Histogram(
                                    x=markowitz_results['final_values'],
                                    name='Markowitz Max Sharpe',
                                    opacity=0.7,
                                    nbinsx=50,
                                    marker_color='blue'
                                ))
                                
                                fig.add_trace(go.Histogram(
                                    x=beta_results['final_values'],
                                    name=beta_strategy['name'],
                                    opacity=0.7,
                                    nbinsx=50,
                                    marker_color='red'
                                ))
                                
                                # Add vertical lines for medians
                                fig.add_vline(
                                    x=markowitz_stats['median_final_value'],
                                    line_dash="dash",
                                    line_color="blue",
                                    annotation_text=f"Markowitz Median: ${markowitz_stats['median_final_value']:,.0f}"
                                )
                                
                                fig.add_vline(
                                    x=beta_stats['median_final_value'],
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Beta Median: ${beta_stats['median_final_value']:,.0f}"
                                )
                                
                                fig.update_layout(
                                    title="Final Portfolio Value Distribution Comparison",
                                    xaxis_title="Final Portfolio Value ($)",
                                    yaxis_title="Frequency",
                                    barmode='overlay',
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Risk metrics comparison
                                st.markdown("### 📉 **Risk Analysis Comparison**")
                                
                                risk_col1, risk_col2 = st.columns(2)
                                
                                with risk_col1:
                                    st.markdown("**Markowitz Maximum Sharpe**")
                                    
                                    # Standard deviation of final values
                                    st.metric("Standard Deviation", f"${pd.Series(markowitz_results['final_values']).std():,.0f}")
                                    
                                    # Downside deviation: only returns below initial investment
                                    downside_values = [v for v in markowitz_results['final_values'] if v < initial_investment]
                                    total_sims = len(markowitz_results['final_values'])
                                    loss_rate = len(downside_values) / total_sims * 100
                                    
                                    if len(downside_values) == 0:
                                        st.metric("Downside Deviation", f"0% simulations lost money")
                                    elif len(downside_values) == 1:
                                        single_loss_pct = (downside_values[0] - initial_investment) / initial_investment * 100
                                        st.metric("Downside Deviation", f"1 sim lost {abs(single_loss_pct):.1f}%")
                                    else:
                                        downside_returns = [(v - initial_investment) / initial_investment for v in downside_values]
                                        downside_dev = pd.Series(downside_returns).std() * initial_investment
                                        st.metric("Downside Deviation", f"${downside_dev:,.0f} ({loss_rate:.1f}% lost)")
                                    
                                    # Maximum drawdown: largest peak-to-trough decline
                                    max_drawdown = 0
                                    for path in markowitz_results['simulation_paths']:
                                        peak = initial_investment
                                        for value in path[1:]:  # Skip initial value
                                            if value > peak:
                                                peak = value
                                            drawdown = (value - peak) / peak
                                            if drawdown < max_drawdown:
                                                max_drawdown = drawdown
                                    st.metric("Maximum Drawdown", f"{max_drawdown*100:.1f}%")
                                
                                with risk_col2:
                                    st.markdown(f"**{beta_strategy['name']}**")
                                    
                                    # Standard deviation of final values
                                    st.metric("Standard Deviation", f"${pd.Series(beta_results['final_values']).std():,.0f}")
                                    
                                    # Downside deviation: only returns below initial investment
                                    downside_values = [v for v in beta_results['final_values'] if v < initial_investment]
                                    total_sims = len(beta_results['final_values'])
                                    loss_rate = len(downside_values) / total_sims * 100
                                    
                                    if len(downside_values) == 0:
                                        st.metric("Downside Deviation", f"0% simulations lost money")
                                    elif len(downside_values) == 1:
                                        single_loss_pct = (downside_values[0] - initial_investment) / initial_investment * 100
                                        st.metric("Downside Deviation", f"1 sim lost {abs(single_loss_pct):.1f}%")
                                    else:
                                        downside_returns = [(v - initial_investment) / initial_investment for v in downside_values]
                                        downside_dev = pd.Series(downside_returns).std() * initial_investment
                                        st.metric("Downside Deviation", f"${downside_dev:,.0f} ({loss_rate:.1f}% lost)")
                                    
                                    # Maximum drawdown: largest peak-to-trough decline
                                    max_drawdown = 0
                                    for path in beta_results['simulation_paths']:
                                        peak = initial_investment
                                        for value in path[1:]:  # Skip initial value
                                            if value > peak:
                                                peak = value
                                            drawdown = (value - peak) / peak
                                            if drawdown < max_drawdown:
                                                max_drawdown = drawdown
                                    st.metric("Maximum Drawdown", f"{max_drawdown*100:.1f}%")
                            
                            with mc_tab4:
                                st.markdown("### 📋 **Complete Statistical Analysis**")
                                
                                # Create comprehensive statistics comparison
                                detailed_stats = {
                                    'Statistic': [
                                        'Initial Investment',
                                        'Time Horizon (Years)',
                                        'Number of Simulations',
                                        '',  # Separator
                                        'RETURNS ANALYSIS',
                                        'Mean Final Value',
                                        'Median Final Value',
                                        'Standard Deviation',
                                        'Mean Annual Return',
                                        'Median Annual Return',
                                        'Mean CAGR',
                                        'Median CAGR',
                                        '',  # Separator
                                        'RISK ANALYSIS',
                                        'Annual Volatility',
                                        'Sharpe Ratio',
                                        'Best Case (95th %ile)',
                                        'Worst Case (5th %ile)',
                                        'Probability of Loss',
                                        'Value at Risk (5%)',
                                        'Expected Shortfall',
                                        '',  # Separator
                                        'PERCENTILE ANALYSIS',
                                        '5th Percentile',
                                        '25th Percentile',
                                        '50th Percentile (Median)',
                                        '75th Percentile',
                                        '95th Percentile'
                                    ],
                                    'Markowitz Max Sharpe': [
                                        f"${initial_investment:,.0f}",
                                        str(time_horizon),
                                        f"{num_simulations:,}",
                                        '',
                                        '',
                                        f"${markowitz_stats['mean_final_value']:,.0f}",
                                        f"${markowitz_stats['median_final_value']:,.0f}",
                                        f"${pd.Series(markowitz_results['final_values']).std():,.0f}",
                                        f"{markowitz_stats['portfolio_expected_return']*100:.2f}%",
                                        f"{((markowitz_stats['median_final_value']/initial_investment)**(1/time_horizon)-1)*100:.2f}%",
                                        f"{markowitz_stats['mean_cagr']*100:.2f}%",
                                        f"{markowitz_stats['median_cagr']*100:.2f}%",
                                        '',
                                        '',
                                        f"{markowitz_stats['portfolio_volatility']*100:.2f}%",
                                        f"{(markowitz_stats['portfolio_expected_return'] - ef_risk_free_rate) / markowitz_stats['portfolio_volatility']:.3f}",
                                        f"${markowitz_stats['best_case']:,.0f}",
                                        f"${markowitz_stats['worst_case']:,.0f}",
                                        f"{markowitz_stats['prob_loss']*100:.1f}%",
                                        f"${markowitz_stats['percentiles']['5th']:,.0f}",
                                        f"${pd.Series([v for v in markowitz_results['final_values'] if v <= markowitz_stats['percentiles']['5th']]).mean():,.0f}",
                                        '',
                                        '',
                                        f"${markowitz_stats['percentiles']['5th']:,.0f}",
                                        f"${pd.Series(markowitz_results['final_values']).quantile(0.25):,.0f}",
                                        f"${markowitz_stats['median_final_value']:,.0f}",
                                        f"${pd.Series(markowitz_results['final_values']).quantile(0.75):,.0f}",
                                        f"${markowitz_stats['percentiles']['95th']:,.0f}"
                                    ],
                                    f"{beta_strategy['name']}": [
                                        f"${initial_investment:,.0f}",
                                        str(time_horizon),
                                        f"{num_simulations:,}",
                                        '',
                                        '',
                                        f"${beta_stats['mean_final_value']:,.0f}",
                                        f"${beta_stats['median_final_value']:,.0f}",
                                        f"${pd.Series(beta_results['final_values']).std():,.0f}",
                                        f"{beta_stats['portfolio_expected_return']*100:.2f}%",
                                        f"{((beta_stats['median_final_value']/initial_investment)**(1/time_horizon)-1)*100:.2f}%",
                                        f"{beta_stats['mean_cagr']*100:.2f}%",
                                        f"{beta_stats['median_cagr']*100:.2f}%",
                                        '',
                                        '',
                                        f"{beta_stats['portfolio_volatility']*100:.2f}%",
                                        f"{(beta_stats['portfolio_expected_return'] - ef_risk_free_rate) / beta_stats['portfolio_volatility']:.3f}",
                                        f"${beta_stats['best_case']:,.0f}",
                                        f"${beta_stats['worst_case']:,.0f}",
                                        f"{beta_stats['prob_loss']*100:.1f}%",
                                        f"${beta_stats['percentiles']['5th']:,.0f}",
                                        f"${pd.Series([v for v in beta_results['final_values'] if v <= beta_stats['percentiles']['5th']]).mean():,.0f}",
                                        '',
                                        '',
                                        f"${beta_stats['percentiles']['5th']:,.0f}",
                                        f"${pd.Series(beta_results['final_values']).quantile(0.25):,.0f}",
                                        f"${beta_stats['median_final_value']:,.0f}",
                                        f"${pd.Series(beta_results['final_values']).quantile(0.75):,.0f}",
                                        f"${beta_stats['percentiles']['95th']:,.0f}"
                                    ]
                                }
                                
                                detailed_df = pd.DataFrame(detailed_stats)
                                st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                                
                                # Export comparison results
                                st.markdown("**📥 Export Comparison Results:**")
                                
                                export_col1, export_col2 = st.columns(2)
                                
                                with export_col1:
                                    comparison_csv = comparison_df.to_csv(index=False)
                                    st.download_button(
                                        label="📊 Download Comparison Summary",
                                        data=comparison_csv,
                                        file_name=f"portfolio_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with export_col2:
                                    detailed_csv = detailed_df.to_csv(index=False)
                                    st.download_button(
                                        label="📋 Download Detailed Statistics",
                                        data=detailed_csv,
                                        file_name=f"detailed_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )
                        
                        else:
                            if markowitz_status != "success":
                                st.error(f"❌ Markowitz simulation failed: {markowitz_status}")
                            if beta_status != "success":
                                st.error(f"❌ Beta strategy simulation failed: {beta_status}")
                
                # Single portfolio simulation (fallback if beta strategy not available)
                if not beta_strategy_available:
                    st.markdown("### 📊 Single Portfolio Simulation")
                    st.info("💡 Run both 'Generate Efficient Frontier' and 'Find Optimal Beta Strategy' to enable head-to-head comparison")
                    
                    if st.button("🎲 Run Markowitz Simulation Only", type="secondary"):
                        # Ensure we have the required data from session state
                        if (st.session_state.ef_results is None or 
                            st.session_state.ef_annual_returns is None or 
                            st.session_state.ef_cov_matrix is None):
                            st.error("❌ Please run 'Generate Efficient Frontier' first to get the required portfolio data.")
                        else:
                            # Get data from session state
                            optimization_results = st.session_state.ef_results
                            annual_returns = st.session_state.ef_annual_returns
                            cov_matrix = st.session_state.ef_cov_matrix
                            max_sharpe = optimization_results['max_sharpe_portfolio']
                            
                            with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations..."):
                                mc_results, mc_status = run_monte_carlo_simulation(
                                    max_sharpe['weights'],
                                    annual_returns,
                                    cov_matrix,
                                    initial_investment=initial_investment,
                                    time_horizon=time_horizon,
                                    num_simulations=num_simulations
                                )
                                
                                if mc_status == "success":
                                    st.success("✅ Monte Carlo simulation completed!")
                                elif mc_status != "success":
                                    st.error(f"❌ Monte Carlo simulation failed: {mc_status}")
                                
                                if mc_status == "success":
                                    # Define portfolio name for display
                                    portfolio_name = "Maximum Sharpe Portfolio"
                                    
                                    # Display single portfolio results (existing code)
                                    mc_tab1, mc_tab2, mc_tab3 = st.tabs([
                                        "📈 Simulation Paths", 
                                        "📊 Final Value Distribution", 
                                        "📋 Statistics Summary"
                                    ])
                                
                                    # [Keep existing single portfolio display code here]
                                    with mc_tab1:
                                        st.markdown("**Portfolio Simulation Paths: Maximum Sharpe Portfolio**")
                                        simulation_plot = create_monte_carlo_plot(mc_results, "Maximum Sharpe Portfolio")
                                        if simulation_plot:
                                            st.plotly_chart(simulation_plot, use_container_width=True)
                                    
                                    # [Add other tabs as needed]
                                    # Key insights
                                    mc_stats = mc_results['statistics']
                                    st.markdown("**📌 Key Insights:**")
                                    
                                    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                                    
                                    with insight_col1:
                                        st.metric(
                                            "Expected Final Value",
                                            f"${mc_stats['mean_final_value']:,.0f}",
                                            f"{((mc_stats['mean_final_value']/initial_investment)**((1/time_horizon))-1)*100:+.1f}% CAGR"
                                        )
                                    
                                    with insight_col2:
                                        st.metric(
                                            "Median Final Value",
                                            f"${mc_stats['median_final_value']:,.0f}",
                                            f"{((mc_stats['median_final_value']/initial_investment)**((1/time_horizon))-1)*100:+.1f}% CAGR"
                                        )
                                    
                                    with insight_col3:
                                        st.metric(
                                            "Probability of Loss",
                                            f"{mc_stats['prob_loss']*100:.1f}%",
                                            delta=None
                                        )
                                    
                                    with insight_col4:
                                        value_at_risk = mc_stats['percentiles']['5th']
                                        st.metric(
                                            "5% Value-at-Risk",
                                            f"${value_at_risk:,.0f}",
                                            f"{((value_at_risk/initial_investment)-1)*100:+.1f}%"
                                    )
                                
                                with mc_tab2:
                                    st.markdown(f"**Distribution of Final Portfolio Values: {portfolio_name}**")
                                    distribution_plot = create_monte_carlo_distribution_plot(mc_results, portfolio_name)
                                    if distribution_plot:
                                        st.plotly_chart(distribution_plot, use_container_width=True)
                                    
                                    # Percentile analysis
                                    st.markdown("**📊 Percentile Analysis:**")
                                    percentile_data = {
                                        'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                                        'Final Value': [
                                            f"${mc_stats['percentiles']['5th']:,.0f}",
                                            f"${pd.Series(mc_results['final_values']).quantile(0.25):,.0f}",
                                            f"${mc_stats['median_final_value']:,.0f}",
                                            f"${pd.Series(mc_results['final_values']).quantile(0.75):,.0f}",
                                            f"${mc_stats['percentiles']['95th']:,.0f}"
                                        ],
                                        'Total Return': [
                                            f"{((mc_stats['percentiles']['5th']/initial_investment)-1)*100:+.1f}%",
                                            f"{((pd.Series(mc_results['final_values']).quantile(0.25)/initial_investment)-1)*100:+.1f}%",
                                            f"{((mc_stats['median_final_value']/initial_investment)-1)*100:+.1f}%",
                                            f"{((pd.Series(mc_results['final_values']).quantile(0.75)/initial_investment)-1)*100:+.1f}%",
                                            f"{((mc_stats['percentiles']['95th']/initial_investment)-1)*100:+.1f}%"
                                        ]
                                    }
                                    percentile_df = pd.DataFrame(percentile_data)
                                    st.dataframe(percentile_df, use_container_width=True, hide_index=True)
                                
                                with mc_tab3:
                                    st.markdown(f"**Complete Statistics Summary: {portfolio_name}**")
                                    
                                    # Create statistics table
                                    stats_table = create_monte_carlo_statistics_table(mc_results, portfolio_name)
                                    if stats_table is not None:
                                        st.dataframe(stats_table, use_container_width=True, hide_index=True)
                                    
                                    # Export Monte Carlo results
                                    st.markdown("**📥 Export Monte Carlo Results:**")
                                    
                                    export_col1, export_col2 = st.columns(2)
                                    
                                    with export_col1:
                                        # Create simulation summary for export
                                        mc_summary_data = {
                                            'Portfolio_Type': ["Maximum Sharpe"],
                                            'Initial_Investment': [initial_investment],
                                            'Time_Horizon_Years': [time_horizon],
                                            'Number_of_Simulations': [num_simulations],
                                            'Expected_Annual_Return_%': [mc_stats['portfolio_expected_return'] * 100],
                                            'Annual_Volatility_%': [mc_stats['portfolio_volatility'] * 100],
                                            'Mean_Final_Value': [mc_stats['mean_final_value']],
                                            'Median_Final_Value': [mc_stats['median_final_value']],
                                            'Best_Case': [mc_stats['best_case']],
                                            'Worst_Case': [mc_stats['worst_case']],
                                            'Probability_of_Loss_%': [mc_stats['prob_loss'] * 100],
                                            'Value_at_Risk_5%': [mc_stats['percentiles']['5th']],
                                            'Value_at_Risk_95%': [mc_stats['percentiles']['95th']],
                                            'Mean_CAGR_%': [mc_stats['mean_cagr'] * 100],
                                            'Median_CAGR_%': [mc_stats['median_cagr'] * 100]
                                        }
                                        
                                        mc_summary_df = pd.DataFrame(mc_summary_data)
                                        mc_summary_csv = mc_summary_df.to_csv(index=False)
                                        
                                        st.download_button(
                                            label="📊 Download MC Summary",
                                            data=mc_summary_csv,
                                            file_name=f"monte_carlo_summary_max_sharpe_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                            mime="text/csv"
                                        )
                                    
                                    with export_col2:
                                        # Create detailed simulation data for export (sample of paths)
                                        sample_paths = mc_results['simulation_paths'][:100]  # First 100 paths
                                        
                                        paths_df = pd.DataFrame(sample_paths.T, 
                                                              columns=[f'Simulation_{i+1}' for i in range(len(sample_paths))])
                                        paths_df.index = [f'Year_{i}' for i in range(time_horizon + 1)]
                                        paths_csv = paths_df.to_csv()
                                        
                                        st.download_button(
                                            label="📈 Download Sample Paths",                                            data=paths_csv,
                                            file_name=f"monte_carlo_paths_max_sharpe_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                            mime="text/csv"
                                        )
                
                # Information section
                with st.expander("ℹ️ About Efficient Frontier Analysis"):
                    st.markdown("""
                    **Modern Portfolio Theory (Markowitz) & CAPM Integration**
                    
                    The Efficient Frontier represents the set of optimal portfolios offering the highest expected return 
                    for each level of risk. Key concepts:
                    
                    - **Diversification**: Combining assets to reduce overall portfolio risk
                    - **Risk-Return Trade-off**: Higher returns typically require accepting higher risk
                    - **Sharpe Ratio**: Measures risk-adjusted returns (higher is better)
                    - **Minimum Variance Portfolio**: Lowest risk portfolio on the frontier
                    - **Tangency Portfolio**: Highest Sharpe ratio (optimal risk-adjusted returns)
                    
                    **Beta Analysis & CAPM Integration:**
                    - **Beta**: Measures stock sensitivity to market movements (S&P 500 benchmark)
                    - **Beta = 1.0**: Moves with the market
                    - **Beta > 1.0**: More volatile than market (higher risk/reward)
                    - **Beta < 1.0**: Less volatile than market (lower risk/reward)
                    - **Portfolio Beta**: Weighted average of individual stock betas
                    - **CAPM Expected Return**: E(R) = Rf + β(Rm - Rf) using 10% market return assumption
                    - **R-Squared**: How well stock movements correlate with market (higher = more reliable beta)
                    
                    **Enhanced Analysis Features:**
                    - **Up to 50 stocks** supported for comprehensive analysis
                    - **Individual stock betas** calculated vs S&P 500
                    - **Optimal portfolio betas** for minimum variance and maximum Sharpe portfolios
                    - **CAPM vs Historical** return comparison for each stock
                    
                    **Interpretation:**
                    - Points on the efficient frontier are optimal portfolios
                    - Points below the frontier are sub-optimal (same risk, lower return)
                    - Points above the frontier are not achievable with these assets
                    - The Capital Allocation Line shows the best risk-return combinations when risk-free investing is allowed
                    
                    **Custom Date Range Analysis:**
                    - Uses your selected start/end months for historical analysis
                    - Minimum 24 months recommended for reliable optimization
                    - Results depend heavily on the chosen time period
                    
                    **Portfolio Integration:**
                    - "Your Portfolio Holdings" automatically loads individual stocks from your actual portfolio
                    - Excludes index funds and ETFs for more meaningful individual stock analysis
                    
                    **Limitations:**
                    - Based on historical data (past performance doesn't guarantee future results)
                    - Assumes normal distribution of returns
                    - Transaction costs and taxes not considered
                    - Beta calculations require sufficient historical data overlap with market
                    - CAPM model assumes market efficiency and single-factor risk model
                    """)
        
        elif not date_range_valid:
            st.error("❌ Please fix the date range selection above")
        else:
            st.info("💡 **Getting Started**: Select stocks above and ensure valid date range to generate the efficient frontier analysis")

if __name__ == "__main__":
    main()