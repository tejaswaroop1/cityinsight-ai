import os
import pandas as pd

folder = "data"

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    try:
        if file.lower().endswith(".csv"):
            df = pd.read_csv(path, encoding="latin1", nrows=5)

        elif file.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(path, nrows=5)

        else:
            print("Skipped")
            continue

        print("Columns:")
        print(list(df.columns))

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print("ERROR:", e)  