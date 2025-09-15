import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
    driver_acceptance_rate,
    wait_time,
    wait_time_squared,
    distance_time_interaction
)
from src.utils.store import AssignmentStore
from haversine import haversine


def main():
    store = AssignmentStore()
    dataset = store.get_submission("dataset.csv")
    dataset = apply_feature_engineering(dataset)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    store.put_submission("train_data.csv", train_data)
    store.put_submission("test_data_split.csv", test_data)
    test_data = store.get_submission("test_data_processed.csv")
    driver_counts = store.get_submission("driver_counts.csv")
    acceptance_rate = store.get_submission("acceptance_rate.csv")
    test_data = apply_feature_engineering_test(test_data, driver_counts, acceptance_rate, train_data)
    store.put_submission("test_data.csv", test_data)
    

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(driver_historical_completed_bookings)
        .pipe(driver_acceptance_rate)
        .pipe(wait_time)
        .pipe(wait_time_squared)
        .pipe(distance_time_interaction)
    )
    scaler = StandardScaler()
    numerical_features = [
        'driver_distance',
        'historical_completed_bookings',
        'driver_acceptance_rate',
        'wait_time',
        'wait_time_squared',
        'distance_time_interaction'
    ]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    store = AssignmentStore()
    driver_counts = df[df['booking_status'] == 'COMPLETED'].groupby('driver_id').size().reset_index(name='historical_completed_bookings')
    acceptance_rate = df.groupby('driver_id')['target'].mean().reset_index(name='driver_acceptance_rate')
    store.put_submission("driver_counts.csv", driver_counts)
    store.put_submission("acceptance_rate.csv", acceptance_rate)
    return df

def apply_feature_engineering_test(df: pd.DataFrame, driver_counts: pd.DataFrame, acceptance_rate: pd.DataFrame, train_data: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to test data."""
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ) if pd.notna(r["driver_latitude"]) and pd.notna(r["pickup_latitude"]) else 0,
        axis=1,
    )
    df["event_hour"] = df["event_timestamp_participant"].apply(
        lambda x: pd.to_datetime(x).hour if pd.notna(x) else 0
    )
    df = df.merge(driver_counts, on='driver_id', how='left')
    df['historical_completed_bookings'] = df['historical_completed_bookings'].fillna(0)
    df = df.merge(acceptance_rate, on='driver_id', how='left')
    df['driver_acceptance_rate'] = df['driver_acceptance_rate'].fillna(train_data['driver_acceptance_rate'].mean())
    df['wait_time'] = train_data['wait_time'].mean()
    df['wait_time_squared'] = train_data['wait_time_squared'].mean()
    df['distance_time_interaction'] = train_data['distance_time_interaction'].mean()
    scaler = StandardScaler()
    numerical_features = [
        'driver_distance',
        'historical_completed_bookings',
        'driver_acceptance_rate',
        'wait_time',
        'wait_time_squared',
        'distance_time_interaction'
    ]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

if __name__ == "__main__":
    main()