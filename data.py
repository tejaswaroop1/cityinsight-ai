import pandas as pd


def load_datasets():

    data = {}

    data["smart_city"] = pd.read_csv(
        "data/smart_city.csv.csv",
        encoding="latin1"
    )

    data["finance"] = pd.read_csv(
        "data/finance.csv",
        encoding="latin1"
    )

    data["cyber"] = pd.read_csv(
        "data/cyber.csv",
        encoding="latin1"
    )

    data["social"] = pd.read_csv(
        "data/social_media.csv",
        encoding="latin1"
    )

    data["environment"] = pd.read_csv(
        "data/air_quality.csv",
        encoding="latin1"
    )

    data["disaster"] = pd.read_csv(
        "data/rainfall.csv",
        encoding="latin1"
    )

    return data