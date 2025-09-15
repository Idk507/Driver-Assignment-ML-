import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from src.utils.store import AssignmentStore

def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    timestamp_col = 'event_timestamp' 
    df = df.dropna(subset=['order_id', 'booking_status'])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.rename(columns={timestamp_col: 'event_timestamp_booking'})
    return df

def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    timestamp_col = 'event_timestamp'
    df = df.dropna(subset=['order_id', 'participant_status'])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.rename(columns={timestamp_col: 'event_timestamp_participant'})
    return df

def clean_test_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    timestamp_col = 'event_timestamp'
    df = df.dropna(subset=['order_id'])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.rename(columns={timestamp_col: 'event_timestamp_participant'})
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target'] = (df['participant_status'] == 'ACCEPTED').astype(int)
    return df

def main():
    store = AssignmentStore()
    booking_df = store.get_raw('booking_log.csv')
    participant_df = store.get_raw('participant_log.csv')
    test_data = store.get_raw('test_data.csv')
    booking_df = clean_booking_df(booking_df)
    participant_df = clean_participant_df(participant_df)
    test_data = clean_test_data(test_data)
    dataset = participant_df.merge(
        booking_df,
        on='order_id',
        how='left',
        suffixes=('_participant', '_booking')
    )
    dataset = dataset.rename(columns={'driver_id_participant': 'driver_id'})
    dataset = dataset.drop(columns=['driver_id_booking'], errors='ignore')
    dataset = create_target(dataset)
    store.put_submission('dataset.csv', dataset)
    store.put_submission('test_data_processed.csv', test_data)

if __name__ == "__main__":
    main()