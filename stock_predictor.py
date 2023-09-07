import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# constants
START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


# functions
@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


@st.cache_resource
def load_news(ticker):
    data = yf.Ticker(ticker)
    return data.news


def pluralize(count, word):
    if count > 1:
        return f"{word}s"
    return word


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    fig.update_layout(xaxis_title="Time", yaxis_title="Stock Price")
    st.plotly_chart(fig)


def compute_volatility(data):
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    volatility = data["Log_Return"].std() * np.sqrt(252)  # Annualize daily volatility
    return volatility


def trend_insight(data):
    # Checking if the latest close price is lower than the average of the last 50 days
    if data["Close"].tail(1).values[0] < data["Close"].tail(180).mean():
        return "The stock is currently in a downward trend compared to the last 50 days average."
    else:
        return "The stock is currently in an upward trend compared to the last 50 days average."


def get_news_sentiment(headlines):
    
    if len(headlines) == 0:
        return "There is no news data."

    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(news)["compound"] for news in headlines]
    avg_sentiment = sum(sentiments) / len(sentiments)

    if avg_sentiment > 0.05:
        return "The recent news sentiment about the stock is positive. 😀"
    elif avg_sentiment < -0.05:
        return "The recent news sentiment about the stock is negative. ☹️"
    else:
        return "The recent news sentiment about the stock is neutral. 😐"


# header
st.title("Equivision")

stock_name = st.text_area("Enter stock name")


# caption for selection box
st.caption("Some stock names to try:")
st.caption(
    "GOOG = Google, APPL = APPLE, MSFT = Microsoft, GME = Gamestop, TSLA = Tesla, MO = Altria, BABA = Alibaba, NVDA = NVIDIA"
)
st.caption(
    "NYA = NYSE Composite , VTI = Vanguard Total Stock Market ETF, VOO = Vanguard 500 Index Fund"
)

# slider
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365
st.markdown("---")  # Add vertical spacing

# status indicator
data_load_state = st.text("Loading data...")

try:
    # load data from yahoo finance
    data = load_data(stock_name)
    if stock_name == "" or len(data) == 0:
        raise ValueError

    # change indicator status
    data_load_state.text("Loading data... done!")

    # show data
    st.subheader("Financial Data from Year 2017")
    st.write(data.head())
    st.write(data.tail())
    st.markdown("---")
    st.subheader("Financial Data from Year 2017 Plotted")
    plot_raw_data()
    st.markdown("---")

    # Filter out data
    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # put data into model and predict
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot predictions
    st.subheader(f"Forecast data for {n_years} {pluralize(n_years, 'year')}")
    st.write(forecast.head())
    st.write(forecast.tail())
    st.markdown("---")
    st.subheader(f"Forecast plot for {n_years} {pluralize(n_years, 'year')}")
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(xaxis_title="Time", yaxis_title="Stock Price")
    st.plotly_chart(fig1)
    st.markdown("---")
    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # calculate volatility and trend
    volatility = compute_volatility(data)
    trend = trend_insight(data)
    predicted_start = forecast["yhat"].iloc[0]
    predicted_end = forecast["yhat"].iloc[-1]

    if predicted_end > predicted_start:
        prediction_trend = "The stock is predicted to increase over the forecasted period. Consider purchasing this stock and holding for the forecasted period."
    else:
        prediction_trend = "The stock is predicted to decrease over the forecasted period. Consider selling the stock now."

    # comment on volatility and trend
    st.markdown("---")
    st.subheader("Risk Insights")
    st.write(f"Volatility: {volatility:.2%}")
    st.write(trend)
    st.write(prediction_trend)

    st.markdown("---")
    st.subheader("Headlines")

    # get news
    news = load_news(stock_name)
    headlines = []
    for article in news:
        st.text(article["title"])
        headlines.append(article["title"])

    st.markdown("---")
    st.subheader("Sentiment Analysis")
    news_sentiment = get_news_sentiment(headlines)
    st.write(news_sentiment)

except:
    data_load_state.text("No data fetched. Invalid input.")


# Comments

# # selection box for stocks
# selected_stock = st.selectbox("Select dataset for prediction", STOCKS)
# STOCKS = (
#     "GOOG",
#     "AAPL",
#     "MSFT",
#     "GME",
#     "TSLA",
#     "MO",
#     "BABA",
#     "NYA",
#     "VTI",
#     "VOO",
#     "NVDA",
# )
