import datetime

import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor


def merge_datasets(financial_calendar_path, market_values_path, output_path):
    # Load the financial calendar and market values datasets
    df1 = pd.read_csv(financial_calendar_path)
    df2 = pd.read_csv(market_values_path)

    # Convert 'Date' column in df1 and df2 to datetime format, setting invalid parsing to NaT
    df1['date'] = pd.to_datetime(df1['date'], errors='coerce').dt.date
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce').dt.date
    df2.set_index('date', inplace=True)


    # Create the 'gov_shutdown' column to indicate the government shutdown period
    shutdown_start_date = pd.to_datetime('2018-12-22')
    shutdown_end_date = pd.to_datetime('2019-01-25')
    df2['gov_shutdown'] = (df2['date'] >= shutdown_start_date) & (df2['date'] <= shutdown_end_date)

    merged_df = pd.merge(df1, df2, on='date', how='outer')

    # Save the merged dataset
    #merged_df.to_csv(output_path, index=False)
    return merged_df


if __name__ == "__main__":
    financial_calendar_path = 'pivoted_finantial_calendar.csv'
    market_values_path = 'market_values.csv'
    output_path = 'merged_financial_calendar_data.csv'

    df = merge_datasets(financial_calendar_path, market_values_path, output_path)
    # df.drop(columns='Unnamed: 0', inplace=True)

    # Assume df is your DataFrame
    # Group data by symbol
    grouped = df.groupby('Symbol')

    # Initialize a DataFrame to store feature importances for each symbol
    feature_importances = pd.DataFrame()

    # Initialize time recording
    total_symbols = len(grouped)
    processed_symbols = 0
    start_time = time.time()

    for name, group in grouped:
        iter_start_time = time.time()
        # Prepare data
        X = group.drop(columns=['Close', 'Symbol'])
        X['date'] = X['date'].map(datetime.date.toordinal)
        # X = X.select_dtypes(exclude=['date'])  # Exclude datetime columns
        y = group['Close']

        # Handle NaNs
        y = y.dropna()
        X = X.loc[y.index]
        X = X.fillna(0)

        # RandomForest for feature selection
        rf = RandomForestRegressor(n_estimators=50)
        rf.fit(X, y)

        # Store feature importance
        feature_importances[name] = pd.Series(rf.feature_importances_, index=X.columns)

        # To avoid highly fragmented dataset
        # Initialize an empty DataFrame with predefined columns
        feature_importances = pd.DataFrame(index=X.columns, columns=grouped.groups.keys())

        # Then fill in the DataFrame within your loop
        feature_importances.loc[:, name] = pd.Series(rf.feature_importances_, index=X.columns)

        # Time estimation
        processed_symbols += 1
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        total_time = iter_end_time - start_time
        estimated_time_left = (total_symbols - processed_symbols) * (total_time / processed_symbols)

        print(f"Processed {processed_symbols}/{total_symbols}. Estimated time left: {estimated_time_left:.2f} seconds")

    # Aggregate feature importances across all symbols
    average_importance = feature_importances.mean(axis=1).sort_values(ascending=False)
    top_avg_features = average_importance.index[:10]  # Top 10 features

