import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Stock Investment Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark/light mode compatibility
st.markdown("""
<style>
 /* Main headers - always visible */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #0066cc;
    margin-bottom: 1rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #0066cc;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #0066cc;
    padding-bottom: 0.5rem;
}

/* Information boxes - light background for visibility */
.explanation-box {
    background-color: #ffffff;
    color: #333333;
    padding: 1.2rem;
    border-radius: 12px;
    border: 2px solid #0066cc;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,102,204,0.1);
}

.info-text {
    background-color: #f8f9fa;
    color: #333333;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #0066cc;
    margin: 0.8rem 0;
    font-size: 0.95rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Color coding for predictions - high contrast */
.profit-green {
    color: #ffffff;
    font-weight: bold;
    background-color: #004d00;
    padding: 2px 6px;
    border-radius: 4px;
}

.loss-red {
    color: #ffffff;
    font-weight: bold;
    background-color: #660000;
    padding: 2px 6px;
    border-radius: 4px;
}

.neutral-orange {
    color: #ffffff;
    font-weight: bold;
    background-color: #806e00;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Metric cards - clean design */
.metric-container {
    background-color: #ffffff;
    color: #333333;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

/* Warning/disclaimer box */
.warning-box {
    background-color: #fff8e1;
    color: #e65100;
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 4px solid #ff9800;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(255,152,0,0.1);
}

/* Success message styling */
.success-box {
    background-color: #e8f5e9;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #4caf50;
    margin: 0.5rem 0;
}

/* Stock recommendation boxes */
.recommendation-box {
    background-color: #ffffff;
    color: #333333;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* ENHANCED TABLE STYLING FOR VISIBLE TEXT */
.stDataFrame {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

.stDataFrame > div {
    background-color: #ffffff !important;
}

.stDataFrame table {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-collapse: collapse !important;
    width: 100 !important;
}

/* Table headers - BLACK TEXT */
.stDataFrame thead th {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-bottom: 2px solid #0066cc !important;
    border-right: 1px solid #e0e0e0 !important;
    padding: 12px 8px !important;
    text-align: center !important;
    font-size: 14px !important;
}

/* Table body cells - BLACK TEXT with darker background */
.stDataFrame tbody td {
    background-color: #e0e0e0 !important;
    color: #000000 !important;
    padding: 10px 8px !important;
    border-bottom: 1px solid #c0c0c0 !important;
    border-right: 1px solid #c0c0c0 !important;
    text-align: center !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* Table row styling - BLACK TEXT */
.stDataFrame tbody tr {
    background-color: #e0e0e0 !important;
    color: #000000 !important;
}

/* Hover effects for table rows */
.stDataFrame tbody tr:hover {
    background-color: #c0c0c0 !important;
}

.stDataFrame tbody tr:hover td {
    color: #000000 !important;
}

/* Enhanced colored row styling with BLACK TEXT */
.stDataFrame tbody tr[style*="background-color: #d4edda"] td {
    background-color: #d4edda !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

.stDataFrame tbody tr[style*="background-color: #f8d7da"] td {
    background-color: #f8d7da !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

.stDataFrame tbody tr[style*="background-color: #fff3cd"] td {
    background-color: #fff3cd !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Force BLACK text on all table elements */
.stDataFrame * {
    color: #000000 !important;
}

/* Target specific Streamlit dataframe containers */
div[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
}

div[data-testid="stDataFrame"] table {
    background-color: #ffffff !important;
    color: #000000 !important;
}

div[data-testid="stDataFrame"] th {
    color: #000000 !important;
    background-color: #f0f0f0 !important;
    font-weight: bold !important;
}

div[data-testid="stDataFrame"] td {
    color: #000000 !important;
    background-color: #e0e0e0 !important;
}

/* Additional overrides for any remaining elements */
.element-container .stDataFrame th,
.element-container .stDataFrame td,
.element-container .stDataFrame span,
.element-container .stDataFrame div {
    color: #000000 !important;
}

/* Override any theme-based styling */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] span {
    color: #000000 !important;
}

/* Ensure text is always visible */
.stMarkdown, .stText {
    color: inherit !important;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Button styling */
.stButton > button {
    background-color: #0066cc;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    background-color: #0052a3;
}
</style>
""", unsafe_allow_html=True)
# Header
st.markdown('<h1 class="main-header">💰 Stock Investment Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="explanation-box">
<h3>📋 What is this dashboard?</h3>
This dashboard shows you <b>AI predictions</b> for Indian stock market investments. It tells you which stocks might go <span class="profit-green">UP ⬆️</span>, 
<span class="loss-red">DOWN ⬇️</span>, or stay <span class="neutral-orange">NEUTRAL ➡️</span> tomorrow, along with expected opening and closing prices.
</div>
""", unsafe_allow_html=True)

# File path
excel_file_path = r"filtered_04_08_25_data.xlsx"

try:
    # Read from default path
    with st.spinner("📊 Loading latest stock predictions..."):
        df = pd.read_excel(excel_file_path)
    st.success("✅ Stock data loaded successfully! Ready for analysis.")
except FileNotFoundError:
    st.error(f"❌ Stock data file not found at: {excel_file_path}")
    st.info("Please check the file path.")
    df = None
except Exception as e:
    st.error(f"❌ Error reading the stock data: {str(e)}")
    df = None

if df is not None:
    try:
        # Data preprocessing
        df['Datetime'] = pd.to_datetime(df['Datetime'])

        # Handle missing values in probability columns
        prob_columns = ['UP_Prob', 'DOWN_Prob', 'NEUTRAL_Prob']
        for col in prob_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate expected profit/loss
        df['Expected_Profit_Loss'] = df['Predicted_Close_Price'] - df['Predicted_Open_Price']
        df['Expected_Return_Percent'] = (df['Expected_Profit_Loss'] / df['Predicted_Open_Price']) * 100

        # Sidebar filters
        st.sidebar.markdown("## 🔍 Filter Your Stocks")
        st.sidebar.markdown("Use these filters to find the stocks you're interested in:")

        # Stock filter with "All" option
        all_stocks = sorted(df['STOCK'].unique())

        stock_selection_type = st.sidebar.radio(
            "How do you want to select stocks?",
            ["📊 Show All Stocks", "🎯 Pick Specific Stocks"]
        )

        if stock_selection_type == "📊 Show All Stocks":
            selected_stocks = all_stocks
            st.sidebar.success(f"✅ Showing all {len(all_stocks)} stocks")
        else:
            selected_stocks = st.sidebar.multiselect(
                "Choose the stocks you want to analyze:",
                options=all_stocks,
                default=all_stocks[:10],
                help="Select one or more stocks from the dropdown"
            )

        # Risk tolerance filter
        st.sidebar.markdown("### 🎯 Risk Tolerance")
        confidence_levels = st.sidebar.multiselect(
            "What level of risk are you comfortable with?",
            options=df['Risk_Level'].unique(),
            default=df['Risk_Level'].unique(),
            help="High = More reliable predictions, Medium = Moderate reliability"
        )

        # Investment strategy filter
        st.sidebar.markdown("### 📈 Investment Strategy")
        prediction_types = st.sidebar.multiselect(
            "What type of stocks do you want to see?",
            options=df['Prediction'].unique(),
            default=df['Prediction'].unique(),
            help="UP = Expected to rise, DOWN = Expected to fall, NEUTRAL = No significant change"
        )

        # Confidence range filter
        if 'Confidence' in df.columns:
            st.sidebar.markdown("### 🎲 Prediction Confidence")
            confidence_range = st.sidebar.slider(
                "Minimum confidence level you want:",
                min_value=float(df['Confidence'].min()),
                max_value=float(df['Confidence'].max()),
                value=(float(df['Confidence'].min()), float(df['Confidence'].max())),
                step=0.1,
                help="Higher confidence = More reliable predictions"
            )

        # Filter data based on selections
        if selected_stocks:
            filtered_df = df[
                (df['STOCK'].isin(selected_stocks)) &
                (df['Risk_Level'].isin(confidence_levels)) &
                (df['Prediction'].isin(prediction_types))
                ]
        else:
            filtered_df = pd.DataFrame()

        if 'Confidence' in df.columns and not filtered_df.empty:
            filtered_df = filtered_df[
                (filtered_df['Confidence'] >= confidence_range[0]) &
                (filtered_df['Confidence'] <= confidence_range[1])
                ]

        # Main dashboard
        if not filtered_df.empty:
            # Quick Summary Cards
            st.markdown('<h2 class="sub-header">📊 Quick Investment Summary</h2>', unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_stocks = len(filtered_df)
                st.metric("🏢 Total Stocks", total_stocks, help="Number of stocks in your selection")

            with col2:
                up_predictions = len(filtered_df[filtered_df['Prediction'] == 'UP'])
                st.metric("📈 Expected to Rise", up_predictions,
                          delta=f"{(up_predictions / total_stocks) * 100:.1f}%" if total_stocks > 0 else "0%")

            with col3:
                down_predictions = len(filtered_df[filtered_df['Prediction'] == 'DOWN'])
                st.metric("📉 Expected to Fall", down_predictions,
                          delta=f"{(down_predictions / total_stocks) * 100:.1f}%" if total_stocks > 0 else "0%")

            with col4:
                neutral_predictions = len(filtered_df[filtered_df['Prediction'] == 'NEUTRAL'])
                st.metric("➡️ Neutral Stocks", neutral_predictions,
                          delta=f"{(neutral_predictions / total_stocks) * 100:.1f}%" if total_stocks > 0 else "0%")

            with col5:
                avg_return = filtered_df['Expected_Return_Percent'].mean()
                st.metric("💰 Avg Expected Return", f"{avg_return:.2f}%",
                          delta="Positive" if avg_return > 0 else "Negative")

            # Key insights
            st.markdown("---")
            st.markdown(
                '<div class="info-text">💡 <b>What do these numbers mean?</b><br>• <b>Expected to Rise (UP):</b> AI predicts these stocks will increase in price tomorrow<br>• <b>Expected to Fall (DOWN):</b> AI predicts these stocks will decrease in price tomorrow<br>• <b>Neutral:</b> AI predicts minimal price change<br>• <b>Expected Return:</b> Average percentage gain/loss expected</div>',
                unsafe_allow_html=True)

            # Stock Recommendations Section
            st.markdown('<h2 class="sub-header">🎯 Smart Investment Recommendations</h2>', unsafe_allow_html=True)

            # Best opportunities
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🚀 Best Buying Opportunities")
                st.markdown('<div class="info-text">Stocks predicted to rise with good confidence levels</div>',
                            unsafe_allow_html=True)

                best_buys = filtered_df[
                    (filtered_df['Prediction'] == 'UP') &
                    (filtered_df['Expected_Return_Percent'] > 0)
                    ].nlargest(5, 'Expected_Return_Percent')

                if not best_buys.empty:
                    for idx, row in best_buys.iterrows():
                        profit = row['Expected_Profit_Loss']
                        return_pct = row['Expected_Return_Percent']
                        st.markdown(f"""
                        <div class="recommendation-box">
                        <strong style="color: #00b300; font-size: 1.1em;">{row['STOCK']}</strong> - Expected: <span class="profit-green">+₹{profit:.2f} ({return_pct:.2f}%)</span><br>
                        Opening: <strong>₹{row['Predicted_Open_Price']:.2f}</strong> → Closing: <strong>₹{row['Predicted_Close_Price']:.2f}</strong><br>
                        Risk Level: <em>{row['Risk_Level']}</em>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No strong buying opportunities found with current filters")

            with col2:
                st.markdown("### ⚠️ Stocks to Avoid")
                st.markdown('<div class="info-text">Stocks predicted to fall - consider avoiding or selling</div>',
                            unsafe_allow_html=True)

                avoid_stocks = filtered_df[
                    (filtered_df['Prediction'] == 'DOWN') &
                    (filtered_df['Expected_Return_Percent'] < 0)
                    ].nsmallest(5, 'Expected_Return_Percent')

                if not avoid_stocks.empty:
                    for idx, row in avoid_stocks.iterrows():
                        loss = row['Expected_Profit_Loss']
                        return_pct = row['Expected_Return_Percent']
                        st.markdown(f"""
                        <div class="recommendation-box">
                        <strong style="color: #cc0000; font-size: 1.1em;">{row['STOCK']}</strong> - Expected: <span class="loss-red">₹{loss:.2f} ({return_pct:.2f}%)</span><br>
                        Opening: <strong>₹{row['Predicted_Open_Price']:.2f}</strong> → Closing: <strong>₹{row['Predicted_Close_Price']:.2f}</strong><br>
                        Risk Level: <em>{row['Risk_Level']}</em>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No stocks showing significant downward trends with current filters")

            # Visualizations
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📈 Visual Market Analysis</h2>', unsafe_allow_html=True)

            # Row 1: Market Overview
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🥧 Market Sentiment Overview")
                st.markdown(
                    '<div class="info-text">Shows what percentage of stocks are expected to rise, fall, or stay neutral</div>',
                    unsafe_allow_html=True)

                prediction_counts = filtered_df['Prediction'].value_counts()
                colors = {'UP': '#00cc44', 'DOWN': '#ff4444', 'NEUTRAL': '#ffaa00'}

                fig_pie = px.pie(
                    values=prediction_counts.values,
                    names=prediction_counts.index,
                    title="Market Predictions Distribution",
                    color_discrete_map=colors
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown("#### 📊 Expected Returns Distribution")
                st.markdown('<div class="info-text">Shows how many stocks fall into different profit/loss ranges</div>',
                            unsafe_allow_html=True)

                fig_hist = px.histogram(
                    filtered_df,
                    x='Expected_Return_Percent',
                    color='Prediction',
                    title="Profit/Loss Distribution (%)",
                    nbins=20,
                    color_discrete_map=colors
                )
                fig_hist.update_layout(
                    xaxis_title="Expected Return (%)",
                    yaxis_title="Number of Stocks",
                    height=400
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="black",
                                   annotation_text="Break-even line")
                st.plotly_chart(fig_hist, use_container_width=True)

            # Row 2: Price Analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💰 Opening vs Closing Prices")
                st.markdown(
                    '<div class="info-text">Each dot represents a stock. Points above the line mean closing price > opening price (profit)</div>',
                    unsafe_allow_html=True)

                fig_scatter = px.scatter(
                    filtered_df,
                    x='Predicted_Open_Price',
                    y='Predicted_Close_Price',
                    color='Prediction',
                    hover_data={'STOCK': True, 'Expected_Return_Percent': ':.2f'},
                    title="Opening vs Closing Price Predictions",
                    color_discrete_map=colors
                )

                # Add break-even line
                min_price = min(filtered_df['Predicted_Open_Price'].min(), filtered_df['Predicted_Close_Price'].min())
                max_price = max(filtered_df['Predicted_Open_Price'].max(), filtered_df['Predicted_Close_Price'].max())
                fig_scatter.add_shape(
                    type="line",
                    x0=min_price, y0=min_price,
                    x1=max_price, y1=max_price,
                    line=dict(color="gray", width=2, dash="dash")
                )
                fig_scatter.add_annotation(
                    x=max_price * 0.8, y=max_price * 0.9,
                    text="Break-even line<br>(No profit/loss)",
                    showarrow=True
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                st.markdown("#### 🎯 Risk vs Reward Analysis")
                st.markdown(
                    '<div class="info-text">Shows the relationship between prediction confidence and expected returns</div>',
                    unsafe_allow_html=True)

                if 'Confidence' in filtered_df.columns:
                    fig_risk_reward = px.scatter(
                        filtered_df,
                        x='Confidence',
                        y='Expected_Return_Percent',
                        color='Prediction',
                        size=abs(filtered_df['Expected_Return_Percent']),
                        hover_data={'STOCK': True},
                        title="Confidence vs Expected Returns",
                        color_discrete_map=colors
                    )
                    fig_risk_reward.update_layout(
                        xaxis_title="Prediction Confidence",
                        yaxis_title="Expected Return (%)",
                        height=400
                    )
                    fig_risk_reward.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_risk_reward, use_container_width=True)
                else:
                    st.info("Confidence data not available for risk analysis")

            # Stock Price Details Table
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📋 Detailed Stock Information</h2>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-text">Complete details of all stocks with opening prices, closing prices, and expected profits/losses</div>',
                unsafe_allow_html=True)

            # Prepare detailed table
            detailed_df = filtered_df.copy()
            detailed_df = detailed_df.sort_values('Expected_Return_Percent', ascending=False)

            # Create a more readable table
            display_df = detailed_df[['STOCK', 'Prediction', 'Risk_Level', 'Predicted_Open_Price',
                                      'Predicted_Close_Price', 'Expected_Profit_Loss',
                                      'Expected_Return_Percent']].copy()

            # Add confidence if available
            if 'Confidence' in detailed_df.columns:
                display_df['Confidence'] = detailed_df['Confidence']

            # Rename columns for better understanding
            display_df.columns = ['Stock Name', 'AI Prediction', 'Risk Level', 'Opening Price (₹)',
                                  'Closing Price (₹)', 'Expected Profit/Loss (₹)', 'Expected Return (%)'] + \
                                 (['Confidence Level'] if 'Confidence' in detailed_df.columns else [])


            # Format the display
            def format_prediction(val):
                if val == 'UP':
                    return '📈 BUY/HOLD'
                elif val == 'DOWN':
                    return '📉 SELL/AVOID'
                else:
                    return '➡️ NEUTRAL'


            def format_return(val):
                if val > 0:
                    return f"+{val:.2f}%"
                else:
                    return f"{val:.2f}%"


            def format_profit_loss(val):
                if val > 0:
                    return f"+₹{val:.2f}"
                else:
                    return f"₹{val:.2f}"


            # Apply formatting
            display_df['AI Prediction'] = display_df['AI Prediction'].apply(format_prediction)
            display_df['Expected Return (%)'] = display_df['Expected Return (%)'].apply(format_return)
            display_df['Expected Profit/Loss (₹)'] = display_df['Expected Profit/Loss (₹)'].apply(format_profit_loss)
            display_df['Opening Price (₹)'] = display_df['Opening Price (₹)'].apply(lambda x: f"₹{x:.2f}")
            display_df['Closing Price (₹)'] = display_df['Closing Price (₹)'].apply(lambda x: f"₹{x:.2f}")

            if 'Confidence Level' in display_df.columns:
                display_df['Confidence Level'] = display_df['Confidence Level'].apply(lambda x: f"{x:.1f}%")


            # Color coding function for the entire row
            def highlight_rows(row):
                if '📈 BUY/HOLD' in str(row.values):
                    return ['background-color: #d4edda'] * len(row)  # Light green
                elif '📉 SELL/AVOID' in str(row.values):
                    return ['background-color: #f8d7da'] * len(row)  # Light red
                else:
                    return ['background-color: #fff3cd'] * len(row)  # Light yellow


            styled_table = display_df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_table, use_container_width=True, height=400)

            # Investment Summary
            st.markdown("---")
            st.markdown('<h2 class="sub-header">💡 Investment Summary & Tips</h2>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 📊 Portfolio Insights")
                total_expected_return = filtered_df['Expected_Return_Percent'].sum()
                positive_stocks = len(filtered_df[filtered_df['Expected_Return_Percent'] > 0])
                negative_stocks = len(filtered_df[filtered_df['Expected_Return_Percent'] < 0])

                st.write(f"**Total Stocks Analyzed:** {len(filtered_df)}")
                st.write(f"**Potentially Profitable:** {positive_stocks}")
                st.write(f"**Potentially Loss-making:** {negative_stocks}")
                st.write(f"**Average Expected Return:** {filtered_df['Expected_Return_Percent'].mean():.2f}%")

                if total_expected_return > 0:
                    st.markdown('<div class="success-box">🎉 Overall positive outlook for your selected stocks!</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="warning-box">⚠️ Overall negative outlook - consider reviewing your selection</div>',
                        unsafe_allow_html=True)

            with col2:
                st.markdown("#### 🎯 Risk Analysis")
                high_risk_count = len(filtered_df[filtered_df['Risk_Level'] == 'High'])
                medium_risk_count = len(filtered_df[filtered_df['Risk_Level'] == 'Medium'])

                st.write(f"**High Risk Stocks:** {high_risk_count}")
                st.write(f"**Medium Risk Stocks:** {medium_risk_count}")

                if 'Confidence' in filtered_df.columns:
                    avg_confidence = filtered_df['Confidence'].mean()
                    st.write(f"**Average Confidence:** {avg_confidence:.1f}%")

                    if avg_confidence > 50:
                        st.markdown('<div class="success-box">✅ High confidence predictions</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ Lower confidence - be cautious</div>',
                                    unsafe_allow_html=True)

            with col3:
                st.markdown("#### 💰 Price Ranges")
                min_open = filtered_df['Predicted_Open_Price'].min()
                max_open = filtered_df['Predicted_Open_Price'].max()
                min_close = filtered_df['Predicted_Close_Price'].min()
                max_close = filtered_df['Predicted_Close_Price'].max()

                st.write(f"**Lowest Opening Price:** ₹{min_open:.2f}")
                st.write(f"**Highest Opening Price:** ₹{max_open:.2f}")
                st.write(f"**Lowest Closing Price:** ₹{min_close:.2f}")
                st.write(f"**Highest Closing Price:** ₹{max_close:.2f}")

            # Important disclaimers
            st.markdown("---")
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Important Investment Disclaimers</h4>
            <ul style="color: #e65100; margin-left: 1rem;">
            <li><b>AI Predictions:</b> These are computer-generated forecasts and may not always be accurate</li>
            <li><b>Market Risk:</b> All investments carry risk - never invest more than you can afford to lose</li>
            <li><b>Do Your Research:</b> Always research stocks independently before making investment decisions</li>
            <li><b>Diversify:</b> Don't put all your money in one stock - spread your investments</li>
            <li><b>Past Performance:</b> Historical data doesn't guarantee future results</li>
            <li><b>Consult Professionals:</b> Consider talking to a financial advisor for personalized advice</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("🔍 No stocks match your current filter criteria. Try adjusting the filters in the sidebar.")
            st.info("💡 **Tip:** Try selecting 'Show All Stocks' or changing your risk/prediction filters")

    except Exception as e:
        st.error(f"❌ Error analyzing the stock data: {str(e)}")
        st.info("Please ensure your data file has the correct format with all required columns.")

else:
    st.error("📁 Unable to load stock data. Please check the file path and try again.")

    # Show expected format
    st.markdown("### 📋 Expected Data Format")
    st.info("""
    Your Excel file should contain these columns:
    - STOCK: Stock symbol/name
    - Datetime: Prediction date
    - Confidence: Prediction confidence level
    - Risk_Level: High/Medium/Low
    - Prediction: UP/DOWN/NEUTRAL
    - UP_Prob, DOWN_Prob, NEUTRAL_Prob: Probability percentages
    - Predicted_Open_Price: Expected opening price
    - Predicted_Close_Price: Expected closing price

    """)







