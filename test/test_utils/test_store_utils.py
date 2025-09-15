import unittest
import os
import pandas as pd
import pickle
import json
import pytest
from sklearn.linear_model import LogisticRegression
from src.utils.store import Store, InvalidExtension

class TestStoreUtils(unittest.TestCase):
    def tearDown(self):
        for f in ("test.csv", "test.json", "test.pkl"):
            if os.path.isfile(f):
                os.remove(f)

    def test_store_get_failures(self):
        with pytest.raises(InvalidExtension):
            Store().get_csv("test.txt")
        with pytest.raises(InvalidExtension):
            Store().get_json("test.txt")
        with pytest.raises(InvalidExtension):
            Store().get_pkl("test.txt")

        with pytest.raises(FileNotFoundError):
            Store().get_csv("test.csv")
        with pytest.raises(FileNotFoundError):
            Store().get_json("test.json")
        with pytest.raises(FileNotFoundError):
            Store().get_pkl("test.pkl")

    def test_store_put_failures(self):
        with pytest.raises(InvalidExtension):
            Store().put_csv("test.txt", None)
        with pytest.raises(InvalidExtension):
            Store().put_json("test.txt", None)
        with pytest.raises(InvalidExtension):
            Store().put_pkl("test.txt", None)

        with pytest.raises(TypeError):
            Store().put_csv("test.csv", None)
        with pytest.raises(TypeError):
            Store().put_json("test.json", None)
        with pytest.raises(TypeError):
            Store().put_pkl("test.pkl", None)

    def test_get_and_put_dataframe(self):
        want = pd.DataFrame({"test": [1, 2, 3]})
        Store().put_csv("test.csv", want)
        got = Store().get_csv("test.csv")
        pd.testing.assert_frame_equal(got, want)

    def test_get_and_put_model(self):
        model = LogisticRegression()
        model.fit(
            [[0.1] for _ in range(500)] + [[0.9] for _ in range(500)],
            [1 for _ in range(500)] + [0 for _ in range(500)],
        )
        want = model.predict([[0.9], [0.9], [0.9]])

        Store().put_pkl("test.pkl", model)
        got = Store().get_pkl("test.pkl").predict([[0.9], [0.9], [0.9]])

        self.assertEqual(got.tolist(), want.tolist())

    def test_get_and_put_dict(self):
        want = {"auc": 0.9}
        Store().put_json("test.json", want)
        got = Store().get_json("test.json")
        self.assertEqual(got, want)