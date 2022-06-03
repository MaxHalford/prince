import pathlib
import pandas as pd


DATASETS = pathlib.Path(__file__).parent / "datasets"


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
        pd.read_csv(DATASETS / "per-capita-energy-stacked.csv")
        .query("Year == @year")
        .drop(columns=["Code", "Year"])
        .rename(columns={"Entity": "Country"})
        .rename(columns=lambda x: x.replace(" per capita (kWh)", "").lower())
        .set_index("country")
        .drop(index={"Africa", "Europe", "North America", "World"})
    )
    if normalize:
        return df.div(df.sum(axis="columns"), axis="rows")
    return df


def load_decathlon():
    """The Decathlon dataset from FactoMineR."""
    decathlon = pd.read_csv(DATASETS / "decathlon.csv")
    decathlon.columns = ["athlete", *map(str.lower, decathlon.columns[1:])]
    decathlon.athlete = decathlon.athlete.apply(str.title)
    decathlon = decathlon.set_index(["competition", "athlete"])
    return decathlon


def load_french_elections_2022():
    """Voting data for the 2022 French elections, by region.

    The [original dataset](https://www.data.gouv.fr/fr/datasets/resultats-du-premier-tour-de-lelection-presidentielle-2022-par-commune-et-par-departement/#resources)
    has been transformed into a contingency matrix. The latter tallies the number of votes for the
    12 candidates across all 18 regions. The number of blank and abstentions are also recorded.
    More information about these regions, including a map, can be found
    [on Wikipedia](https://www.wikiwand.com/fr/Région_française).

    """
    dataset = pd.read_csv(DATASETS / "02-resultats-par-region.csv")
    cont = dataset.pivot("reg_name", "cand_nom", "cand_nb_voix")
    cont["Abstention"] = dataset.groupby("reg_name")["abstention_nb"].min()
    cont["Blank"] = dataset.groupby("reg_name")["blancs_nb"].min()
    cont.columns = [c.title() for c in cont.columns]
    cont.index.name = "region"
    cont.columns.name = "candidate"
    return cont
