import unittest
import pandas as pd
import pytest
from src.features.transformations import driver_distance_to_pickup, hour_of_day

class TestTransformationFeatures(unittest.TestCase):
    def test_driver_distance_to_pickup(self):
        df = pd.DataFrame(
            {
                "driver_latitude": [-6.5],
                "driver_longitude": [103.5],
                "pickup_latitude": [-6.5],
                "pickup_longitude": [103.3],
            }
        )
        got = driver_distance_to_pickup(df)
        self.assertIn("driver_distance", got.columns)
        self.assertEqual(len(got.columns), 5)
        self.assertAlmostEqual(got["driver_distance"].tolist()[0], 22.096060298188256)

        with pytest.raises(TypeError):
            driver_distance_to_pickup(df.astype(str))

    def test_hour_of_day(self):
        df = pd.DataFrame({"event_timestamp_participant": ["2015-05-12 05:25:23.904 UTC"]})
        got = hour_of_day(df)
        self.assertIn("event_hour", got.columns)
        self.assertEqual(len(got.columns), 2)
        self.assertEqual(got["event_hour"].tolist(), [5])

    def test_transform_with_invalid_key(self):
        df = pd.DataFrame({"invalid_key": [1, 2, 3]})
        with pytest.raises(KeyError):
            driver_distance_to_pickup(df)
        with pytest.raises(KeyError):
            hour_of_day(df)