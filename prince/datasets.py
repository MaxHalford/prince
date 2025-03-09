from __future__ import annotations

import pathlib

import pandas as pd

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"


def load_energy_mix(year=2019, normalize=True):
    """Per capita energy mix by country in 2019.

    Each row corresponds to a country. There is one column for each energy source.
    A value corresponds to the average energy consumption of a source per capita.
    For instance, in France, every citizen consumed 15,186 kWh of nuclear energy.

    This data comes from https://ourworldindata.org/energy-mix

    Parameters
    ----------
    year
        The year the study was made.
    normalize
        Whether or not to normalize the kWh by country.

    """

    df = (
        pd.read_csv(DATASETS_DIR / "per-capita-energy-stacked.csv")
        .query("Year == @year")
        .query("Entity not in ['Africa', 'Europe', 'North America', 'World']")
        .drop(columns=["Code", "Year"])
        .rename(columns={"Entity": "Country"})
        .rename(columns=lambda x: x.replace(" per capita (kWh)", "").lower())
        .set_index(["continent", "country"])
    )
    if normalize:
        return df.div(df.sum(axis="columns"), axis="rows")
    return df


def load_decathlon():
    """The Decathlon dataset from FactoMineR."""
    decathlon = pd.read_csv(DATASETS_DIR / "decathlon.csv")
    decathlon.columns = ["athlete", *map(str.lower, decathlon.columns[1:])]
    decathlon.athlete = decathlon.athlete.apply(str.title)
    decathlon = decathlon.set_index(["competition", "athlete"])
    return decathlon


def load_french_elections():
    """Voting data for the 2022 French elections, by region.

    The [original dataset](https://www.data.gouv.fr/fr/datasets/resultats-du-premier-tour-de-lelection-presidentielle-2022-par-commune-et-par-departement/#resources)
    has been transformed into a contingency matrix. The latter tallies the number of votes for the
    12 candidates across all 18 regions. The number of blank and abstentions are also recorded.
    More information about these regions, including a map, can be found
    [on Wikipedia](https://www.wikiwand.com/fr/Région_française).

    """
    dataset = pd.read_csv(DATASETS_DIR / "02-resultats-par-region.csv")
    cont = dataset.pivot(index="reg_name", columns="cand_nom", values="cand_nb_voix")
    cont["Abstention"] = dataset.groupby("reg_name")["abstention_nb"].min()
    cont["Blank"] = dataset.groupby("reg_name")["blancs_nb"].min()
    cont.columns = [c.title() for c in cont.columns]
    cont.index.name = "region"
    cont.columns.name = "candidate"
    return cont


def load_punctuation_marks():
    """Punctuation marks of six French writers."""
    return pd.read_csv(DATASETS_DIR / "punctuation_marks.csv", index_col="author")


def load_hearthstone_cards():
    """Hearthstone standard cards.

    Source: https://gist.github.com/MaxHalford/32ed2c80672d7391ec5b4e6f291f14c1

    """
    return pd.read_csv(DATASETS_DIR / "hearthstone_cards.csv", index_col="id")


def load_burgundy_wines():
    """Burgundy wines dataset.

    Source: https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

    """
    wines = pd.DataFrame(
        data=[
            ["Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No"],
            ["No", "Maybe", "Yes", "No", "Yes", "Maybe", "Yes", "No", "Yes", "Yes"],
            ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes"],
            ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            ["Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No"],
            ["Yes", "Maybe", "No", "Yes", "No", "Maybe", "No", "Yes", "No", "No"],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("Expert 1", "Fruity"),
                ("Expert 1", "Woody"),
                ("Expert 1", "Coffee"),
                ("Expert 2", "Red fruit"),
                ("Expert 2", "Roasted"),
                ("Expert 2", "Vanillin"),
                ("Expert 2", "Woody"),
                ("Expert 3", "Fruity"),
                ("Expert 3", "Butter"),
                ("Expert 3", "Woody"),
            ],
            names=("expert", "aspect"),
        ),
        index=[f"Wine {i + 1}" for i in range(6)],
    )
    wines.insert(0, "Oak type", [1, 2, 2, 2, 1, 1])
    return wines


def load_beers():
    """Beers dataset.

    The data is taken from https://github.com/philipperemy/beer-dataset.

    """
    return pd.read_csv(DATASETS_DIR / "beers.csv.zip", index_col="name")


def load_premier_league():
    """Premier League dataset.

    The data is taken from Wikipedia, using pd.read_html.

    """
    return pd.read_csv(DATASETS_DIR / "premier_league.csv", index_col=0, header=[0, 1])
