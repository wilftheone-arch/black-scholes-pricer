"""
Enhanced Black-Scholes Option Pricer
=====================================
A robust option pricing application with:
- Black-Scholes model implementation
- Interactive Streamlit GUI
- P&L Heatmap visualization
- SQLite database persistence
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import uuid
import pandas as pd

# =============================================================================
# BLACK-SCHOLES MODEL
# =============================================================================

def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """Calculate d1 and d2 parameters for Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.

    Parameters:
    -----------
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to expiry (in years)
    r : float - Risk-free interest rate (decimal)
    sigma : float - Volatility (decimal)

    Returns:
    --------
    float - Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes put option price.

    Parameters:
    -----------
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to expiry (in years)
    r : float - Risk-free interest rate (decimal)
    sigma : float - Volatility (decimal)

    Returns:
    --------
    float - Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)."""
    if T <= 0:
        return {"delta_call": 1 if S > K else 0, "delta_put": -1 if S < K else 0,
                "gamma": 0, "theta_call": 0, "theta_put": 0, "vega": 0,
                "rho_call": 0, "rho_put": 0}

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)

    # Delta
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2))
    theta_put = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))

    # Vega (same for call and put, expressed per 1% change)
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100

    # Rho (per 1% change)
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta_call": delta_call, "delta_put": delta_put,
        "gamma": gamma,
        "theta_call": theta_call / 365, "theta_put": theta_put / 365,  # Daily theta
        "vega": vega, "rho_call": rho_call, "rho_put": rho_put
    }


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def init_database():
    """Initialize SQLite database with inputs and outputs tables."""
    conn = sqlite3.connect('mnt/option_calculations.db')
    cursor = conn.cursor()

    # Create inputs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inputs (
            calculation_id TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            stock_price REAL NOT NULL,
            strike_price REAL NOT NULL,
            time_to_expiry REAL NOT NULL,
            volatility REAL NOT NULL,
            interest_rate REAL NOT NULL,
            call_purchase_price REAL,
            put_purchase_price REAL
        )
    ''')

    # Create outputs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            calculation_id TEXT NOT NULL,
            volatility_shock REAL NOT NULL,
            stock_price_shock REAL NOT NULL,
            call_value REAL NOT NULL,
            put_value REAL NOT NULL,
            call_pnl REAL,
            put_pnl REAL,
            FOREIGN KEY (calculation_id) REFERENCES inputs(calculation_id)
        )
    ''')

    conn.commit()
    conn.close()


def save_calculation(calc_id: str, inputs: dict, heatmap_data: list):
    """Save calculation inputs and outputs to database."""
    conn = sqlite3.connect('mnt/option_calculations.db')
    cursor = conn.cursor()

    # Insert inputs
    cursor.execute('''
        INSERT INTO inputs (calculation_id, stock_price, strike_price,
                           time_to_expiry, volatility, interest_rate,
                           call_purchase_price, put_purchase_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (calc_id, inputs['stock_price'], inputs['strike_price'],
          inputs['time_to_expiry'], inputs['volatility'], inputs['interest_rate'],
          inputs.get('call_purchase_price'), inputs.get('put_purchase_price')))

    # Insert outputs (heatmap data)
    for row in heatmap_data:
        cursor.execute('''
            INSERT INTO outputs (calculation_id, volatility_shock, stock_price_shock,
                                call_value, put_value, call_pnl, put_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (calc_id, row['vol'], row['price'], row['call'], row['put'],
              row.get('call_pnl'), row.get('put_pnl')))

    conn.commit()
    conn.close()


def get_calculation_history(limit: int = 10) -> pd.DataFrame:
    """Retrieve recent calculation history from database."""
    conn = sqlite3.connect('mnt/option_calculations.db')
    df = pd.read_sql_query(f'''
        SELECT calculation_id, timestamp, stock_price, strike_price,
               time_to_expiry, volatility, interest_rate,
               call_purchase_price, put_purchase_price
        FROM inputs
        ORDER BY timestamp DESC
        LIMIT {limit}
    ''', conn)
    conn.close()
    return df


# =============================================================================
# HEATMAP GENERATION
# =============================================================================

def generate_heatmap_data(base_S: float, K: float, T: float, r: float,
                          base_sigma: float, vol_range: tuple, price_range: tuple,
                          call_purchase: float = None, put_purchase: float = None,
                          grid_size: int = 20) -> dict:
    """
    Generate heatmap data for option values across volatility and stock price ranges.

    Returns dict with matrices for call values, put values, and P&L.
    """
    vol_min, vol_max = vol_range
    price_min, price_max = price_range

    volatilities = np.linspace(vol_min, vol_max, grid_size)
    stock_prices = np.linspace(price_min, price_max, grid_size)

    call_values = np.zeros((grid_size, grid_size))
    put_values = np.zeros((grid_size, grid_size))
    call_pnl = np.zeros((grid_size, grid_size))
    put_pnl = np.zeros((grid_size, grid_size))

    heatmap_records = []

    for i, sigma in enumerate(volatilities):
        for j, S in enumerate(stock_prices):
            call_val = black_scholes_call(S, K, T, r, sigma)
            put_val = black_scholes_put(S, K, T, r, sigma)

            call_values[i, j] = call_val
            put_values[i, j] = put_val

            c_pnl = call_val - call_purchase if call_purchase else None
            p_pnl = put_val - put_purchase if put_purchase else None

            call_pnl[i, j] = c_pnl if c_pnl is not None else 0
            put_pnl[i, j] = p_pnl if p_pnl is not None else 0

            heatmap_records.append({
                'vol': sigma, 'price': S,
                'call': call_val, 'put': put_val,
                'call_pnl': c_pnl, 'put_pnl': p_pnl
            })

    return {
        'volatilities': volatilities,
        'stock_prices': stock_prices,
        'call_values': call_values,
        'put_values': put_values,
        'call_pnl': call_pnl,
        'put_pnl': put_pnl,
        'records': heatmap_records
    }


def create_pnl_heatmap(data: dict, option_type: str, show_pnl: bool = True) -> go.Figure:
    """Create a Plotly heatmap figure for option values or P&L."""
    volatilities = data['volatilities']
    stock_prices = data['stock_prices']

    if show_pnl:
        z_data = data[f'{option_type}_pnl']
        title = f'{option_type.upper()} Option P&L Heatmap'
        colorscale = [
            [0, 'rgb(180, 0, 0)'],      # Dark red for most negative
            [0.3, 'rgb(255, 80, 80)'],   # Light red
            [0.5, 'rgb(255, 255, 255)'], # White at zero
            [0.7, 'rgb(80, 200, 80)'],   # Light green
            [1, 'rgb(0, 150, 0)']        # Dark green for most positive
        ]
        # Center colorscale at zero
        max_abs = max(abs(z_data.min()), abs(z_data.max()))
        zmin, zmax = -max_abs, max_abs
    else:
        z_data = data[f'{option_type}_values']
        title = f'{option_type.upper()} Option Value Heatmap'
        colorscale = 'Viridis'
        zmin, zmax = None, None

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=stock_prices,
        y=volatilities * 100,  # Convert to percentage
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hovertemplate=(
            'Stock Price: $%{x:.2f}<br>'
            'Volatility: %{y:.1f}%<br>'
            'Value: $%{z:.2f}<extra></extra>'
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Stock Price ($)',
        yaxis_title='Volatility (%)',
        height=500,
        font=dict(size=12)
    )

    return fig


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Black-Scholes Option Pricer",
        page_icon="📈",
        layout="wide"
    )

    # Initialize database
    init_database()

    # Header
    st.title("📈 Enhanced Black-Scholes Option Pricer")
    st.markdown("""
    A comprehensive option pricing tool with interactive visualization and data persistence.
    Enter your parameters below to calculate option prices and visualize P&L scenarios.
    """)

    st.divider()

    # ==========================================================================
    # INPUT SECTION
    # ==========================================================================

    st.header("📊 Option Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Base Inputs")
        stock_price = st.number_input(
            "Stock Price ($)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Current price of the underlying stock"
        )
        strike_price = st.number_input(
            "Strike Price ($)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Strike price of the option"
        )
        time_to_expiry = st.number_input(
            "Time to Expiry (Years)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Time until option expiration in years"
        )

    with col2:
        st.subheader("Market Inputs")
        volatility = st.number_input(
            "Volatility (%)",
            min_value=1.0,
            max_value=200.0,
            value=20.0,
            step=1.0,
            help="Annualized volatility as a percentage"
        ) / 100  # Convert to decimal

        interest_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.1,
            help="Risk-free interest rate as a percentage"
        ) / 100  # Convert to decimal

    with col3:
        st.subheader("P&L Inputs (Optional)")
        call_purchase_price = st.number_input(
            "Call Purchase Price ($)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Price paid for the call option (for P&L calculation)"
        )
        put_purchase_price = st.number_input(
            "Put Purchase Price ($)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Price paid for the put option (for P&L calculation)"
        )

    st.divider()

    # ==========================================================================
    # HEATMAP RANGE CONFIGURATION
    # ==========================================================================

    st.header("🎯 Heatmap Configuration")

    hm_col1, hm_col2 = st.columns(2)

    with hm_col1:
        st.subheader("Stock Price Range")
        price_min = st.number_input(
            "Min Stock Price ($)",
            min_value=0.01,
            value=stock_price * 0.7,
            step=1.0
        )
        price_max = st.number_input(
            "Max Stock Price ($)",
            min_value=price_min + 1,
            value=stock_price * 1.3,
            step=1.0
        )

    with hm_col2:
        st.subheader("Volatility Range")
        vol_min = st.number_input(
            "Min Volatility (%)",
            min_value=1.0,
            max_value=199.0,
            value=max(1.0, volatility * 100 * 0.5),
            step=1.0
        ) / 100
        vol_max = st.number_input(
            "Max Volatility (%)",
            min_value=vol_min * 100 + 1,
            max_value=200.0,
            value=min(200.0, volatility * 100 * 1.5),
            step=1.0
        ) / 100

    st.divider()

    # ==========================================================================
    # CALCULATE BUTTON
    # ==========================================================================

    if st.button("🧮 Calculate Option Prices", type="primary", use_container_width=True):
        # Generate unique calculation ID
        calc_id = str(uuid.uuid4())[:8]

        # Calculate base option prices
        call_price = black_scholes_call(stock_price, strike_price, time_to_expiry,
                                        interest_rate, volatility)
        put_price = black_scholes_put(stock_price, strike_price, time_to_expiry,
                                      interest_rate, volatility)

        # Calculate Greeks
        greeks = calculate_greeks(stock_price, strike_price, time_to_expiry,
                                  interest_rate, volatility)

        # Store in session state
        st.session_state['results'] = {
            'calc_id': calc_id,
            'call_price': call_price,
            'put_price': put_price,
            'greeks': greeks,
            'inputs': {
                'stock_price': stock_price,
                'strike_price': strike_price,
                'time_to_expiry': time_to_expiry,
                'volatility': volatility,
                'interest_rate': interest_rate,
                'call_purchase_price': call_purchase_price if call_purchase_price > 0 else None,
                'put_purchase_price': put_purchase_price if put_purchase_price > 0 else None
            }
        }

        # Generate heatmap data
        heatmap_data = generate_heatmap_data(
            stock_price, strike_price, time_to_expiry, interest_rate, volatility,
            vol_range=(vol_min, vol_max),
            price_range=(price_min, price_max),
            call_purchase=call_purchase_price if call_purchase_price > 0 else None,
            put_purchase=put_purchase_price if put_purchase_price > 0 else None
        )
        st.session_state['heatmap_data'] = heatmap_data

        # Save to database
        save_calculation(calc_id, st.session_state['results']['inputs'],
                        heatmap_data['records'])

        st.success(f"Calculation complete! (ID: {calc_id})")

    # ==========================================================================
    # RESULTS DISPLAY
    # ==========================================================================

    if 'results' in st.session_state:
        results = st.session_state['results']

        st.header("💰 Option Prices")

        price_col1, price_col2, price_col3 = st.columns(3)

        with price_col1:
            st.metric(
                label="Call Option Price",
                value=f"${results['call_price']:.4f}",
                delta=f"P&L: ${results['call_price'] - (results['inputs']['call_purchase_price'] or 0):.4f}"
                      if results['inputs']['call_purchase_price'] else None
            )

        with price_col2:
            st.metric(
                label="Put Option Price",
                value=f"${results['put_price']:.4f}",
                delta=f"P&L: ${results['put_price'] - (results['inputs']['put_purchase_price'] or 0):.4f}"
                      if results['inputs']['put_purchase_price'] else None
            )

        with price_col3:
            moneyness = stock_price / strike_price
            if moneyness > 1.02:
                status = "ITM (Call) / OTM (Put)"
            elif moneyness < 0.98:
                status = "OTM (Call) / ITM (Put)"
            else:
                status = "ATM"
            st.metric(label="Moneyness", value=status)

        st.divider()

        # Greeks Display
        st.header("📐 Option Greeks")
        greeks = results['greeks']

        greek_col1, greek_col2, greek_col3, greek_col4 = st.columns(4)

        with greek_col1:
            st.metric("Delta (Call)", f"{greeks['delta_call']:.4f}")
            st.metric("Delta (Put)", f"{greeks['delta_put']:.4f}")

        with greek_col2:
            st.metric("Gamma", f"{greeks['gamma']:.6f}")
            st.metric("Vega", f"{greeks['vega']:.4f}")

        with greek_col3:
            st.metric("Theta (Call)", f"{greeks['theta_call']:.4f}")
            st.metric("Theta (Put)", f"{greeks['theta_put']:.4f}")

        with greek_col4:
            st.metric("Rho (Call)", f"{greeks['rho_call']:.4f}")
            st.metric("Rho (Put)", f"{greeks['rho_put']:.4f}")

        st.divider()

        # =======================================================================
        # HEATMAP VISUALIZATION
        # =======================================================================

        if 'heatmap_data' in st.session_state:
            st.header("🗺️ P&L Heatmaps")

            heatmap_data = st.session_state['heatmap_data']

            # Toggle for P&L vs Value view
            show_pnl = st.checkbox(
                "Show P&L (requires purchase prices)",
                value=bool(results['inputs']['call_purchase_price'] or
                          results['inputs']['put_purchase_price'])
            )

            hm_col1, hm_col2 = st.columns(2)

            with hm_col1:
                call_fig = create_pnl_heatmap(heatmap_data, 'call', show_pnl)
                st.plotly_chart(call_fig, use_container_width=True)

            with hm_col2:
                put_fig = create_pnl_heatmap(heatmap_data, 'put', show_pnl)
                st.plotly_chart(put_fig, use_container_width=True)

            if show_pnl and (results['inputs']['call_purchase_price'] or
                            results['inputs']['put_purchase_price']):
                st.info("""
                **Interpreting the P&L Heatmaps:**
                - 🟢 **Green regions** indicate positive P&L (profit)
                - 🔴 **Red regions** indicate negative P&L (loss)
                - The heatmaps show how your option P&L changes across different
                  volatility and stock price scenarios
                """)

    st.divider()

    # ==========================================================================
    # CALCULATION HISTORY
    # ==========================================================================

    st.header("📜 Calculation History")

    try:
        history = get_calculation_history(10)
        if not history.empty:
            # Format the dataframe for display
            display_df = history.copy()
            display_df['volatility'] = (display_df['volatility'] * 100).round(2).astype(str) + '%'
            display_df['interest_rate'] = (display_df['interest_rate'] * 100).round(2).astype(str) + '%'
            display_df['stock_price'] = display_df['stock_price'].apply(lambda x: f"${x:.2f}")
            display_df['strike_price'] = display_df['strike_price'].apply(lambda x: f"${x:.2f}")
            display_df['time_to_expiry'] = display_df['time_to_expiry'].apply(lambda x: f"{x:.2f}y")

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No calculations saved yet. Run a calculation to see history.")
    except Exception as e:
        st.warning(f"Could not load history: {e}")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Black-Scholes Model Assumptions:**
    - European-style options only
    - No dividends
    - Constant volatility and interest rates
    - Log-normal distribution of stock prices
    - No transaction costs
    """)


if __name__ == "__main__":
    main()
