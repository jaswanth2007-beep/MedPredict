from src.data import add_rul
import pandas as pd

def test_rul():
    df = pd.DataFrame({"unit":[1,1,1], "cycle":[1,2,3]})
    out = add_rul(df)
    assert out["RUL"].tolist() == [2,1,0]
