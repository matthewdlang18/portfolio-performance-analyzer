# Portfolio Performance Analyzer 📈

A comprehensive Streamlit dashboard for advanced portfolio analysis, optimization, and risk assessment with Modern Portfolio Theory and Monte Carlo simulation capabilities.

![Portfolio Analyzer Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

### **📅 Unlimited Historical Analysis**
- **Any Date Range**: Analyze market performance from **1970 to present day**
- **Dynamic Data Fetching**: Real-time data from Yahoo Finance API
- **Historical Benchmarking**: Compare against decades of S&P 500 performance
- **Era Analysis**: Study different market periods (1970s stagflation, 1980s bull market, dot-com bubble, 2008 crisis, etc.)

### Core Analysis
- **📊 Performance Comparison**: Compare your portfolio against SPX (S&P 500) and SPXE (Equal Weight S&P 500)
- **📈 Time-Weighted Returns**: Properly calculated returns accounting for deposits/withdrawals
- **🎯 Capital Market Line Analysis**: Visualize optimal risk-return trade-offs
- **🏆 Performance Rankings**: See how your portfolio ranks against benchmarks
- **📅 Unlimited Date Ranges**: Analyze performance over ANY time period from **1927 to today** using live Yahoo Finance data
- **📋 Detailed Statistics**: Comprehensive metrics including Sharpe ratios, volatility, and alpha
- **🕰️ Historical Analysis**: Study market behavior across different eras (Great Depression, WWII, 1960s Bull Market, etc.)

### Advanced Portfolio Optimization
- **⚡ Efficient Frontier Analysis**: Modern Portfolio Theory implementation with Markowitz optimization
- **🎯 Portfolio Optimization**: Generate minimum variance and maximum Sharpe ratio portfolios
- **📊 Beta Analysis & CAPM**: Individual stock betas, portfolio beta calculations, and CAPM expected returns
- **🔬 Beta Comparison**: Compare Markowitz vs beta-adjusted portfolio strategies with interactive controls
- **🎲 Monte Carlo Simulation**: Probabilistic portfolio performance forecasting
- **📈 Interactive Visualizations**: Professional Plotly charts with efficient frontier, Capital Allocation Line, and simulation results

### Beta Strategy Analysis ✨ NEW
- **🎯 Target Beta Portfolios**: Adjust portfolio weights to achieve specific market sensitivity (beta)
- **🛡️ Defensive Strategies**: Create low-beta portfolios for conservative investors
- **⚡ Aggressive Strategies**: Build high-beta portfolios for growth-focused investors
- **⚖️ Strategy Comparison**: Side-by-side analysis of Markowitz vs beta-adjusted approaches
- **📊 Visual Weight Analysis**: Interactive charts showing portfolio allocation differences
- **💡 Educational Insights**: Learn why different optimization methods produce different results

### Individual Stock Analysis  
- **📈 Stock Performance**: Compare individual stocks from your portfolio against the S&P 500
- **📊 Beta Calculations**: Stock-specific beta, alpha, and R-squared metrics
- **📈 Correlation Analysis**: Interactive scatterplots showing market correlation
- **🎯 CAPM Integration**: Capital Asset Pricing Model expected return calculations

### Portfolio Holdings Analysis ✨ NEW
- **📂 Holdings Overview**: Complete portfolio positions with current values and performance
- **📊 Portfolio Beta Analysis**: Calculate overall portfolio beta vs S&P 500
- **📈 Portfolio Scatterplot**: Interactive visualization of portfolio-level beta relationship
- **🔍 Individual Position Betas**: Beta analysis for each stock position in your portfolio
- **💰 Portfolio Composition**: Visual breakdown of holdings with pie charts

### Monte Carlo Risk Analysis
- **🎲 Portfolio Simulation**: Run thousands of scenarios to forecast portfolio performance
- **📊 Risk Assessment**: Value-at-Risk calculations and probability distributions
- **📈 Performance Paths**: Visualize potential portfolio trajectories over time
- **📋 Statistical Analysis**: Comprehensive risk metrics and confidence intervals

## Screenshots

### Performance Comparison
The dashboard provides side-by-side comparison of key metrics:
- Annualized/Total Returns
- Volatility (Risk)
- Sharpe Ratios
- Performance Rankings with medal indicators 🥇🥈🥉

### Capital Market Line
Visualize where your portfolio sits relative to the optimal risk-return frontier:
- Shows if your portfolio is above, on, or below the efficient frontier
- Calculates alpha (excess return over market expectations)
- Provides optimal allocation recommendations

### Individual Stock Analysis
Analyze how individual stocks in your portfolio perform against the S&P 500:
- Select any stock from your portfolio holdings
- View monthly returns comparison charts
- Calculate stock-specific beta, alpha, and R-squared metrics
- Interactive scatterplot showing correlation with market movements
- Performance metrics comparison with detailed statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/portfolio-analyzer.git
   cd portfolio-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   ./run_dashboard.sh
   # Or manually: streamlit run portfolio_analyzer.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Data Requirements

The analyzer expects your portfolio data in CSV format with the following structure:

### Investment Income Balance Detail CSV
```csv
Monthly,Beginning balance,Market change,Dividends,Interest,Deposits,Withdrawals,Net advisory fees,Ending balance
Aug 2024,"$100,000.00","$2,500.00","$250.00",$0.00,$0.00,$0.00,$0.00,"$102,750.00"
Jul-24,"$95,000.00","$4,200.00","$300.00",$0.00,"$5,000.00",$0.00,$0.00,"$100,000.00"
```

**Sample file included:** `sample_portfolio_data.csv`

### Key Fields:
- **Monthly**: Date in format "MMM-YY" or "MMM YYYY"
- **Beginning balance**: Portfolio value at start of period
- **Ending balance**: Portfolio value at end of period  
- **Deposits**: Money added to portfolio
- **Withdrawals**: Money removed from portfolio

## Configuration

### Risk-Free Rate
Default: 3.044% (current 10-year Treasury rate)
- Adjustable in the sidebar
- Used for Sharpe ratio and alpha calculations

### Date Ranges
- Full range: **1970 - Present** (leverages Yahoo Finance historical data)
- Customizable start/end dates
- Minimum 2 months of data required

### Return Types
- **Annualized Returns**: Geometric mean annualized
- **Total Returns**: Cumulative return over period

## Technical Details

### Calculations

**Time-Weighted Returns**
```python
return = (ending_balance - beginning_balance - net_deposits) / beginning_balance
```

**Sharpe Ratio**
```python
sharpe = (annualized_return - risk_free_rate) / annualized_volatility
```

**Alpha (vs Capital Market Line)**
```python
expected_return = risk_free_rate + market_sharpe * portfolio_volatility
alpha = portfolio_return - expected_return
```

### Data Sources
- **SPX Data**: Real-time S&P 500 data from Yahoo Finance API (1927-present) with automatic fallback to static data
- **SPXE Data**: Equal Weight S&P 500 ETF data via Yahoo Finance API with static fallback
- **Portfolio Data**: Your investment account statements
- **Individual Stock Data**: Real-time data fetched via Yahoo Finance API for any publicly traded stock
- **Historical Range**: Nearly 100 years of market data available (1927 to today)
- **Live Updates**: Data automatically fetched for your selected date range

## Project Structure

```
portfolio-analyzer/
├── portfolio_analyzer.py      # Main Streamlit application
├── requirements.txt          # Python dependencies
├── run_dashboard.sh         # Launch script
├── README.md               # This file
├── sample_portfolio_data.csv # Sample portfolio data format
└── .venv/                 # Virtual environment (created after setup)
```

## Dependencies

- **streamlit**: Web dashboard framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive charts and visualizations
- **yfinance**: Real-time financial data for individual stock analysis

## 🚀 Quick Start & Deployment

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/portfolio-analyzer.git
   cd portfolio-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run portfolio_analyzer.py
   ```

4. **Open your browser** to `http://localhost:8501`

### 🌐 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Connect your GitHub account** and select this repository

4. **Set the main file path**: `portfolio_analyzer.py`

5. **Deploy!** Your app will be live at `https://your-app-name.streamlit.app`

### 📁 Sample Data

The app includes sample data files:
- `Investment_income_balance_detail.csv` - Sample income/dividend data
- `Portfolio_Positions_Aug-11-2025.csv` - Sample portfolio positions
- `sample_portfolio_data.csv` - Additional sample data

**To use your own data**: Replace these files with your data in the same CSV format, or use the file upload feature in the app.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and informational purposes only. It should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.

## Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/YOUR_USERNAME/portfolio-analyzer/issues) page
2. Create a new issue with detailed description
3. Include error messages and data format examples

---

**Built with ❤️ using Streamlit and Python**
