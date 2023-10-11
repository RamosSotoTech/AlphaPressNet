import investpy
import pandas as pd
from datetime import datetime, timedelta

def gather_economic_calendar(start_date, end_date):
    # Convert string dates to datetime objects
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")

    # Ensure end_date is not earlier than start_date
    if end_date < start_date:
        raise ValueError("end_date should not be earlier than start_date")

    # Convert datetime objects back to string for investpy function call
    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')

    # Fetch the economic calendar data
    econ_calendar = investpy.economic_calendar(time_zone='GMT', time_filter='time_only',
                                               from_date=start_date_str, to_date=end_date_str,
                                               countries=['united states'],
                                               importances=['high', 'medium', 'low'])

    # # Create a dictionary to hold dates and events
    # date_event_dict = {}
    #
    # # Loop through the economic calendar data and populate the dictionary
    # for index, row in econ_calendar.iterrows():
    #     # Convert string to datetime object before formatting it
    #     date = datetime.strptime(row['date'], '%d/%m/%Y').strftime('%Y-%m-%d')
    #     event = row['event']
    #     if date not in date_event_dict:
    #         date_event_dict[date] = [event]
    #     else:
    #         date_event_dict[date].append(event)
    #
    # # Convert the dictionary to a DataFrame
    # dates = []
    # events = []
    # for date, event_list in date_event_dict.items():
    #     dates.append(date)
    #     events.append(', '.join(event_list))
    #
    # dataset = pd.DataFrame({'Date': dates, 'Events': events})

    return econ_calendar
