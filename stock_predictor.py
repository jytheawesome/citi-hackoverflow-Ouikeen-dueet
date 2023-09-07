# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA', 'MO', 'BABA', 'NYA', 'VTI', 'VOO', 'NVDA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
st.caption('GOOG = Google, APPL = APPLE, MSFT = Microsoft, GME = Gamestop, TSLA = Tesla, MO = Altria, BABA = Alibaba, NVDA = NVIDIA')
st.caption('The above stocks are individual company stocks. Investing in them is inherently more risky than composite stocks.')
st.caption('NYA = NYSE Composite , VTI = Vanguard Total Stock Market ETF, VOO = Vanguard 500 Index Fund')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)


def compute_volatility(data):
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    volatility = data['Log_Return'].std() * np.sqrt(252) # Annualize daily volatility
    return volatility


def trend_insight(data):
    # Checking if the latest close price is lower than the average of the last 50 days
    if data['Close'].tail(1).values[0] < data['Close'].tail(180).mean():
        return "The stock is currently in a downward trend compared to the last 50 days average."
    else:
        return "The stock is currently in an upward trend compared to the last 50 days average."

# pip install vaderSentiment
def get_news_sentiment(stock):
    # In a real-world scenario, you would extract recent news articles about the stock.
    # For this PoC, let's assume we have a list of recent news titles about the stock.
    # news_titles = ["This " + stock + "is the best! MUST BUY NOW!",
    #                "This is a dummy neutral news about " + stock]

    news_titles = [user_text]
    
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(news)["compound"] for news in news_titles]
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    if avg_sentiment > 0.05:
        return "The recent news sentiment about the stock is positive. 😀"
    elif avg_sentiment < -0.05:
        return "The recent news sentiment about the stock is negative. ☹️"
    else:
        return "The recent news sentiment about the stock is neutral. 😐"
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

volatility = compute_volatility(data)
trend = trend_insight(data)


st.subheader('Risk Insights')
st.write(f"Volatility: {volatility:.2%}")
st.write(trend)

st.subheader('Sentiment Analysis')
user_text = st.text_area("Enter text for sentiment analysis:")
news_sentiment = get_news_sentiment(selected_stock)
st.write(news_sentiment)

