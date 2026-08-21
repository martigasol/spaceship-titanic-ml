import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering functions to the DataFrame.
    """

    df = create_total_spend(df)
    df = create_has_spent_money(df)
    df = create_group_size(df)
    df = create_is_alone(df)
    df = extract_deck(df)
    df = extract_cabin_side(df)

    return df

SPENDING_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]


def create_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a TotalSpend feature by summing all onboard expenses.
    """

    df = df.copy()

    df["TotalSpend"] = (
        df[SPENDING_COLUMNS]
        .fillna(0)
        .sum(axis=1)
    )

    return df

def create_has_spent_money(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary feature indicating whether
    the passenger spent any money onboard.
    """

    df = df.copy()

    df["HasSpentMoney"] = (
        df["TotalSpend"] > 0
    )

    return df

def create_group_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the GroupSize feature from PassengerId.
    """

    df = df.copy()

    group_id = df["PassengerId"].str.split("_").str[0] #separem en llista i agafem 1er element (0001)

    group_sizes = group_id.value_counts() #contem tots els elements de cada grup

    df["GroupSize"] = group_id.map(group_sizes) #substiuim 0001 per 4 (vegades que apareix 0001)

    return df

def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary feature indicating whether
    the passenger is travelling alone.
    """

    df = df.copy()

    df["IsAlone"] = df["GroupSize"] == 1

    return df

def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the deck from the Cabin feature.
    """

    df = df.copy()

    df["Deck"] = (
        df["Cabin"]
        .str.split("/")
        .str[0]
    )

    return df

def extract_cabin_side(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the cabin side from the Cabin feature.
    """

    df = df.copy()

    df["CabinSide"] = (
        df["Cabin"]
        .str.split("/")
        .str[2]
    )

    return df