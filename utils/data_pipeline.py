import calendar

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
from functools import partial
from dateutil.parser import parse
import warnings
from typing import Optional
from statsmodels.tsa.seasonal import STL

from pandas import DateOffset


def test_patterns(pattern_dict, sample_dict):
    results = {}
    for key, pattern in pattern_dict.items():
        results[key] = [re.sub(pattern, 'Generic ', event) for event in sample_dict[key]]
    return results


# ^FOMC Member [A-Z][a-z]+ Speaks$


def transform_early_close_events(row):
    event_pattern = r"United States - (.+?) - Early close at (\d{2}:\d{2})"
    match = re.match(event_pattern, row['event'])

    if match:
        row['event'] = 'Early Close'
        row['time'] = match.group(2)

    return row


def transform_us_holidays(row):
    holiday_pattern = r"United States - (.+?)"

    if row['time'] == '00:00':
        match = re.match(holiday_pattern, row['event'])
        if match:
            row['event'] = 'Holiday'

    return row


def get_event_containing(value, df):
    return df[df['event'].str.contains(value, case=False, regex=False)].copy()


def replace_event_containing(value, df, replacement):
    df.loc[df['event'].str.contains(value, case=False, regex=False), 'event'] = replacement
    return df.copy()


def replace_with_value(df, column, string_list, replace_value, exact_match=True):
    if exact_match:
        df[column] = df[column].apply(lambda x: replace_value if x in string_list else x)
    else:
        pattern = '|'.join(string_list)
        df.loc[df[column].str.contains(pattern, na=False), column] = replace_value


# Function to identify and remove month abbreviations from events that occur in that specific month

def remove_month_from_event(row, match_month=False):
    if match_month:
        month_str = row['date'].strftime('%b') \
            if isinstance(row['date'], datetime) else row['date'].iloc[0].strftime('%b')
        pattern = r'(' + re.escape(month_str) + r')'
        row['event'] = re.sub(pattern, '', row['event']).strip()
    else:
        row['event'] = re.sub(r'\((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\)', '', row['event']).strip()
    return row.copy()


def remove_periodic_abbreviations(row):
    row['event'] = re.sub(r'\((WoW|QoQ|MoM|YoY)\)', '', row['event']).strip()
    return row.copy()


def remove_quarter_abbreviations(row):
    row['event'] = re.sub(r'\(Q\d\)', '', row['event']).strip()
    return row.copy()


def remove_tentative_time(row):
    if 'Tentative' in str(row['time']):
        return None
    return row


def anonymize_event(event):
    # Regex pattern to match names that start and end with a capital letter and have a space in between.
    # This should match names like "FOMC Member Williams Speaks" and transform it to "FOMC Member Speaks"
    pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'

    # Replace matched pattern with a generic term
    generic_event = re.sub(pattern, 'Member', event)

    return generic_event


# Update the detect_datetime_format function to return only the format strings
def identify_datetime_format(df, date_col='date', time_col='time'):
    """
    Identifies the format of the date and time columns.
    """
    # Check the data type of the date and time columns
    date_dtype = df[date_col].dtype
    time_dtype = df[time_col].dtype

    # If date is already in datetime64 format, no need to identify its format
    if pd.api.types.is_datetime64_any_dtype(date_dtype):
        date_format = None
    else:
        # List of possible date formats
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y', '%d.%m.%Y', '%m.%d.%Y']

        # Identify date format
        date_format = None
        sample_date_str = str(df[date_col].iloc[0])
        for fmt in date_formats:
            try:
                datetime.strptime(sample_date_str, fmt)
                date_format = fmt
                break
            except ValueError:
                continue

    # If time is already in datetime64 or timedelta64 format, no need to identify its format
    if pd.api.types.is_datetime64_any_dtype(time_dtype) or pd.api.types.is_timedelta64_dtype(time_dtype):
        time_format = None
    else:
        # List of possible time formats
        time_formats = ['%H:%M', '%H:%M:%S']

        # Identify time format
        time_format = None
        sample_time_str = str(df[time_col].iloc[0])
        for fmt in time_formats:
            try:
                datetime.strptime(sample_time_str, fmt)
                time_format = fmt
                break
            except ValueError:
                continue

    return date_format, time_format


def create_datetime_column(df, date_col='date', time_col='time'):
    """
    Combines the date and time columns into a new datetime column.
    """
    # Filter out rows with NaN date or time
    df_filtered = df.dropna(subset=[date_col, time_col])

    # Identify the formats of date and time columns
    date_format, time_format = identify_datetime_format(df_filtered, date_col, time_col)

    if date_format and time_format:
        # Combine date and time into datetime
        datetime_format = f"{date_format} {time_format}"
        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str),
                                        format=datetime_format, errors='coerce')
    elif date_format:
        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str),
                                        format=f"{date_format} %H:%M", errors='coerce')
    elif time_format:
        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str),
                                        format=f"%Y-%m-%d {time_format}", errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')

    return df


# Define the transformation function
def generalize_event(event):
    # Define patterns and their respective replacements, in order of restrictiveness
    patterns = [
        (r'FOMC Member .* Speaks', 'FOMC Member Speaks'),
        (r'Chicago Fed President .* Speaks', 'Chicago Fed President Speaks'),
        (r'Fed Chair .* Speaks', 'Fed Chair Speaks'),
        (r'Fed Vice Chair for Supervision .* Speaks', 'Fed Vice Chair for Supervision Speaks'),
        (r'Fed Vice Chair .* Speaks', 'Fed Vice Chair Speaks'),
        (r'U\.S\. President .* Speaks', 'U.S. President Speaks'),
        (r'OPEC Crude [oO]il Production .* \(Barrel\)', 'OPEC Crude Oil Production'),
        (r'Treasury Secretary .* Speaks', 'Treasury Secretary Speaks'),
        (r'Sec .* Speaks', 'Sec Speaks'),
        (r'Fed Governor .* Speaks', 'Fed Governor Speaks'),
        (r'Fed Governor .* Testifies', 'Fed Governor Testifies'),
        (r'Fed .* Speaks', 'Fed Member Speaks'),
        (r'Fed .* Testimony', 'Fed Member Testifies'),
        (r'Fed .* Testifies', 'Fed Member Testifies')
    ]

    # Apply transformations based on the most restrictive match
    for pattern, replacement in patterns:
        if re.search(pattern, event, re.IGNORECASE):
            return re.sub(pattern, replacement, event, flags=re.IGNORECASE)

    return event


def find_periodic_events(df, column_name='event'):
    pattern = r'\((WoW|QoQ|MoM|YoY)\)'
    events_with_pattern = df[df[column_name].str.contains(pattern, regex=True)][column_name].unique()
    unique_events_without_pattern = set()

    for event in events_with_pattern:
        event_without_pattern = re.sub(pattern, '', event).strip()
        unique_events_without_pattern.add(event_without_pattern)

    return sorted(list(unique_events_without_pattern))


def find_month_including_events(df, column_name='event'):
    pattern = r'\((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\)'
    events_with_pattern = df[df[column_name].str.contains(pattern, regex=True)][column_name].unique()
    unique_events_without_pattern = set()

    for event in events_with_pattern:
        event_without_pattern = re.sub(pattern, '', event).strip()
        unique_events_without_pattern.add(event_without_pattern)

    return sorted(list(unique_events_without_pattern))


def add_reported_month_column(df: pd.DataFrame, event_column: str, new_column: str) -> pd.DataFrame:
    """
    Adds a new column to the dataframe that contains the month mentioned in the 'event' column.

    Parameters:
    - df: DataFrame containing the financial data
    - event_column: The name of the column containing event details
    - new_column: The name of the new column that will store the reported month

    Returns:
    - Modified DataFrame with the new column
    """
    month_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'

    def extract_month(event: str) -> Optional[str]:
        match = re.search(month_pattern, event)
        return match.group(0) if match else None

    df[new_column] = df[event_column].apply(extract_month)
    return df


def remove_duplicates_ignore_id(df, ignore_column='id'):
    """
    Remove duplicate rows from a DataFrame, ignoring a specific column.

    Parameters:
        df (DataFrame): The input DataFrame.
        ignore_column (str): The column to ignore when checking for duplicates.

    Returns:
        DataFrame: A new DataFrame with duplicates removed.
    """
    # Drop the ignore_column temporarily for duplicate checking
    df_temp = df.drop(columns=[ignore_column])

    # Find duplicates and remove them
    duplicates = df_temp.duplicated(keep='first')
    df_no_duplicates = df[~duplicates]

    return df_no_duplicates


def filter_percentage_events(df):
    # Use a regular expression to match percentages in the 'actual' column
    df_percentage = df[df['actual'].str.contains('%', na=False)]

    # Get unique events that have percentages
    unique_percentage_events = df_percentage['event'].unique().tolist()

    return unique_percentage_events


def extract_residual(df, event1, event2, new_event_name):
    # Filter out only the rows containing the specified events
    event1_data = df[df['event'] == event1].copy()
    event2_data = df[df['event'] == event2].copy()

    # Clean up the 'actual' values and convert to float
    event1_data['actual'] = event1_data['actual'].str.replace('%', '').str.replace('M', '').str.replace(',', '').astype(float)
    event2_data['actual'] = event2_data['actual'].str.replace('%', '').str.replace('M', '').str.replace(',', '').astype(float)

    # Merge the two event data on the 'datetime' column
    merged_data = pd.merge(event1_data, event2_data, on='datetime', suffixes=(f'_{event1}', f'_{event2}'))

    # Calculate the residual (Event1 - Event2)
    merged_data['residual'] = merged_data[f'actual_{event1}'] - merged_data[f'actual_{event2}']

    # Create a new DataFrame to store the first event and the calculated residual
    residual_data = merged_data[['datetime', 'residual']].copy()
    residual_data['event'] = new_event_name
    residual_data = residual_data[['datetime', 'event', 'residual']]
    residual_data.rename(columns={'residual': 'actual'}, inplace=True)

    # Drop the original second event data and append the new residual data
    df = df[df['event'] != event2]
    df = pd.concat([df, residual_data], ignore_index=True)

    return df


def remove_other_events(df, event_keyword, keep_events):
    # Remove all events containing the keyword except for the ones in keep_events list
    df = df[~df['event'].str.contains(event_keyword, na=False) | df['event'].isin(keep_events)]
    return df


def add_is_report(df):
    df['is_report'] = None

    # Identify events that contain both instances (values None and not-None)
    events_with_both = df.groupby('event').apply(lambda x: x['forecast'].isna().any() and x['forecast'].notna().any())

    # Only populate the new column for events that contain both instances
    for event in events_with_both.index:
        if events_with_both[event]:
            df.loc[df['event'] == event, 'is_report'] = df.loc[df['event'] == event, 'forecast'].notna()

    return df


def calculate_start_date(row):
    event = row['event']
    date = row['date']

    # Regular expression pattern to find periodic string
    pattern = r'\((WoW|QoQ|MoM|YoY)\)'

    # Search for the periodic string in the event
    match = re.search(pattern, event)

    if match:
        periodic_string = match.group(1)
        if periodic_string == "MoM":
            return date - timedelta(days=30)
        elif periodic_string == "WoW":
            return date - timedelta(weeks=1)
        elif periodic_string == "QoQ":
            return date - pd.DateOffset(months=3)
        elif periodic_string == "YoY":
            return date - pd.DateOffset(years=1)
    else:
        return date


# Function to handle the reported_month and align it with start_date for use with apply
def align_reported_date(row):
    start_date = row['start_date']
    reported_month = row['reported_month']

    # Dictionary to convert month names to month numbers
    month_to_num = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    if pd.isna(reported_month) or pd.isna(start_date):
        return start_date

    # Extract the month number from the start_date
    start_month_num = start_date.month

    # Convert reported_month to its numerical representation
    reported_month_num = month_to_num.get(reported_month, None)

    if reported_month_num is None:
        return start_date

    if reported_month_num != start_month_num:
        # Extrapolate the date by adding an additional month
        return start_date + pd.DateOffset(months=1)
    else:
        return start_date


# Function to handle the aggregation process for specific events
def aggregate_event_data(df, event_name, agg_func='sum'):
    event_data = df[df['event'] == event_name].copy()
    aggregated_data = event_data.groupby('date')['actual_numeric'].agg(agg_func).reset_index()
    aggregated_data['event'] = event_name
    df = df[df['event'] != event_name]
    df = pd.concat([df, aggregated_data], ignore_index=True, sort=False)
    return df


def convert_to_numeric(value):
    if pd.isna(value):
        return None

    # Remove commas
    value = value.replace(',', '')

    # Handle percentages
    if "%" in value:
        return float(value.strip('%')) / 100

    # Handle negative numbers
    if "-" in value:
        is_negative = True
        value = value.replace('-', '')
    else:
        is_negative = False

    # Handle suffixes for Billion, Million, Thousand, Trillion
    suffix_multiplier = {
        'B': 1e9,
        'M': 1e6,
        'K': 1e3,
        'T': 1e12
    }
    for suffix, multiplier in suffix_multiplier.items():
        if suffix in value:
            return float(value.replace(suffix, '')) * multiplier * (-1 if is_negative else 1)

    # Handle other cases, assuming they are already numeric
    return float(value) * (-1 if is_negative else 1)


def limited_backward_fill(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    for col in df.columns:
        match = re.search(r'\((WoW|QoQ|MoM|YoY)\)', col)
        if match:
            limit_unit = match.group(1)

            if limit_unit == 'WoW':
                limit_value = pd.DateOffset(weeks=1)
            elif limit_unit == 'QoQ':
                limit_value = pd.DateOffset(months=3)
            elif limit_unit == 'MoM':
                limit_value = pd.DateOffset(months=1)
            elif limit_unit == 'YoY':
                limit_value = pd.DateOffset(years=1)

            last_valid_index = None
            for idx in df.index:
                if pd.notna(df.loc[idx, col]):
                    next_index = idx + limit_value
                    if next_index in df.index:
                        df.loc[last_valid_index:next_index, col] = df.loc[last_valid_index:next_index, col].fillna(
                            method='bfill')
                    last_valid_index = idx
    return df

def load_and_preprocess_data(input_path):
    # Load the initial dataset
    df = pd.read_csv(input_path)

    df.drop(columns=['Unnamed: 0', 'zone', 'currency', 'importance'], inplace=True)
    df = df.drop_duplicates()

    df = df.astype({'id': int, 'time': 'str', 'event': 'str'})

    date_format, time_format = identify_datetime_format(df, 'date', 'time')

    df['date'] = pd.to_datetime(df['date'], format=date_format, errors='coerce')
    df = df[df['time'] != 'Tentative'].copy()

    # Replace "All Day" with "00:00" in the 'time' column of the DataFrame
    df['time'] = df['time'].apply(lambda x: '00:00' if x == 'All Day' else x)

    df = df.apply(transform_early_close_events, axis=1)
    df = df.apply(transform_us_holidays, axis=1)

    df = add_reported_month_column(df, 'event', 'reported_month')
    # Remove month indicators from the 'event' column
    df = df.apply(partial(remove_month_from_event, match_month=False), axis=1)
    df = df.apply(remove_quarter_abbreviations, axis=1)
    # df = add_is_report(df)

    # Calculate start_dates using apply and add as a new column
    df['start_date'] = df.apply(calculate_start_date, axis=1)
    # Apply the function to create the explicit_reported_date column
    df['explicit_reported_date'] = df.apply(align_reported_date, axis=1)

    # Apply the function to each event group and concatenate the results
    # df = pd.concat([fill_missing_values(group) for _, group in df.groupby('event')])

    df = create_datetime_column(df)

    df['event'] = df['event'].apply(generalize_event)

    df = df[~df['event'].str.contains(' s.a', case=False, na=False, regex=False)].copy()
    df = df[~df['event'].str.contains('Speaks', case=False, na=False, regex=False)].copy()
    df = df[~df['event'].str.contains('Testimony', case=False, na=False, regex=False)].copy()
    df = df[~df['event'].str.contains('Testifies', case=False, na=False, regex=False)].copy()
    df = remove_duplicates_ignore_id(df, ignore_column='id')

    df = extract_residual(df.copy(), 'CPI Index', 'Core CPI Index', 'CPI-Core CPI Residual')

    df = extract_residual(df.copy(), 'PPI Index', 'Core PPI Index', 'PPI-Core PPI Residual')
    # Handle PCE data (assuming 'PCE Price Index' and 'Core PCE Price Index' as the two versions)
    df = extract_residual(df.copy(), 'PCE Price Index', 'Core PCE Price Index', 'PCE-Core PCE Residual')
    # Remove other PPI events except the ones specified
    df = remove_other_events(df, 'PPI', ['PPI Index', 'Core PPI Index', 'PPI-Core PPI Residual'])

    # Remove other PCE events except the ones specified
    df = remove_other_events(df, 'PCE', ['PCE Price Index', 'Core PCE Price Index', 'PCE-Core PCE Residual'])

    # Remove other CPI events as an example (assuming 'Core CPI Index' and 'Core CPI Index Residual' are to be kept)
    df = remove_other_events(df, 'CPI', ['Core CPI Index', 'Core CPI Index Residual'])

    df['actual_numeric'] = df['actual'].apply(convert_to_numeric)
    df = aggregate_event_data(df, 'Crude Oil Imports', agg_func='sum')
    df = aggregate_event_data(df, 'OPEC Crude Oil Production', agg_func='sum')

    # Filter events that only contain NaN values
    events_only_nan = df.groupby('event')['actual_numeric'].apply(lambda x: all(pd.isna(x))).reset_index()
    events_only_nan = events_only_nan[events_only_nan['actual_numeric']]

    # Get the unique events that only contain NaN values
    events_with_nan = events_only_nan['event'].unique()
    df.loc[df['event'].isin(events_with_nan), 'actual_numeric'] = True

    df = df.sort_values(by='datetime').drop_duplicates(subset=['date', 'event'], keep='last')

    df_pivoted = df.pivot(index='date', columns='event', values='actual_numeric')
    df_pivoted = limited_backward_fill(df_pivoted)

    auction_list = [
        "10-Year Note Auction",
        "10-Year TIPS Auction",
        "2-Year Note Auction",
        "20-Year Bond Auction",
        "20-Year TIPS Auction",
        "30-Year Bond Auction",
        "30-Year TIPS Auction",
        "4-Week Bill Auction",
        "5-Year Note Auction",
        "5-Year TIPS Auction",
        "52-Week Bill Auction",
        "7-Year Note Auction",
        "8-Week Bill Auction"
    ]

    df_pivoted[auction_list] = df_pivoted[auction_list].fillna(method='ffill')



    # df_new = df[df['date'] >= '2019-06-01'].copy()

    # df = df.apply(remove_periodic_abbreviations, axis=1)
    #
    # df = df[df['importance'] == 'high'].copy()
    #
    # # Convert 'date' column to datetime format
    # df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #
    # # Replace specific event names
    # replacement_dict = {
    #     'Fed Powell Speaks': 'Fed Chair Speaks',
    #     'Fed Powell Yellen Speaks': 'Fed Chair Speaks',
    #     'Sec Kashkari': 'Sec Speaks',
    #     'Trump': 'President',
    #     'Biden': 'President'
    # }
    # df['event'].replace(replacement_dict, inplace=True)
    #
    # # Combine 'date' and 'time' into a single datetime column
    # df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    #
    # # Drop unnecessary columns
    # df.drop(columns=['Unnamed: 0', 'zone', 'currency', 'date', 'time'], inplace=True)
    #
    # # Remove exact duplicates
    # df.drop_duplicates(inplace=True)
    #
    # df = df.pivot(index='datetime', columns='event', values='actual').copy()

    return df


if __name__ == "__main__":
    input_path = "economicCalendar.csv"
    # output_path = "your_output_file_path_here.csv"

    df = load_and_preprocess_data(input_path)
    # df.to_csv(output_path, index=False)
