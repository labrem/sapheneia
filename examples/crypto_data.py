from binance.client import Client
import pandas as pd


def main():
    client = Client(tld='us')

    # Get daily k-lines for ETH-USDT for all of 2023
    klines = client.get_historical_klines(
        symbol='BONKUSDT',
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str="1 Jan, 2025",
        end_str="10 Oct, 2025"
    )

    # The data comes in a specific format, easily converted to a DataFrame
    df = pd.DataFrame(klines, columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    print(df[['timestamp', 'close']].head())

if __name__ == "__main__":
    main()