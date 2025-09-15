import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from haversine import haversine



def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["driver_latitude", "driver_longitude", "pickup_latitude", "pickup_longitude"]
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ) if pd.notna(r["driver_latitude"]) and pd.notna(r["pickup_latitude"]) else 0,
        axis=1,
    )
    return df

def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp_participant"].apply(
        lambda x: pd.to_datetime(x).hour if pd.notna(x) else 0
    )
    return df

def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    completed_bookings = df[df["booking_status"] == "COMPLETED"]
    driver_counts = completed_bookings.groupby("driver_id").size().reset_index(name="historical_completed_bookings")
    
    df = df.merge(driver_counts, on="driver_id", how="left")
    df["historical_completed_bookings"] = df["historical_completed_bookings"].fillna(0)
    
    return df

def driver_acceptance_rate(df: pd.DataFrame) -> pd.DataFrame:
    acceptance_rate = df.groupby("driver_id")["target"].mean().reset_index(name="driver_acceptance_rate")
    df = df.merge(acceptance_rate, on="driver_id", how="left")
    df["driver_acceptance_rate"] = df["driver_acceptance_rate"].fillna(df["driver_acceptance_rate"].mean())
   
    return df

def wait_time(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["event_timestamp_participant", "event_timestamp_booking"]   
    df['event_timestamp_participant'] = pd.to_datetime(df['event_timestamp_participant'], errors='coerce')
    df['event_timestamp_booking'] = pd.to_datetime(df['event_timestamp_booking'], errors='coerce')
    
    df['wait_time'] = (
        df['event_timestamp_participant'] - df['event_timestamp_booking']
    ).dt.total_seconds() / 60.0
    df['wait_time'] = df['wait_time'].fillna(df['wait_time'].mean())
   
    return df

def wait_time_squared(df: pd.DataFrame) -> pd.DataFrame:
    df['wait_time_squared'] = df['wait_time'] ** 2
    df['wait_time_squared'] = df['wait_time_squared'].fillna(df['wait_time_squared'].mean())
    return df

def distance_time_interaction(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["driver_distance", "wait_time"]
    df['distance_time_interaction'] = df['driver_distance'] * df['wait_time']
    df['distance_time_interaction'] = df['distance_time_interaction'].fillna(df['distance_time_interaction'].mean())
    return df