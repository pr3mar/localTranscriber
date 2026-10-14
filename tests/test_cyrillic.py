import pytest

from localtranscribe.cyrillic import CYRILLIC_LANGS, to_cyrillic


def test_cyrillic_languages():
    assert CYRILLIC_LANGS == {"mk", "sr"}


@pytest.mark.parametrize(
    ("latin", "expected"),
    [
        ("Ljubav i njegova džamija", "Љубав и његова џамија"),
        ("Ćao, đače!", "Ћао, ђаче!"),
        ("LJUBAV", "ЉУБАВ"),
        ("Zdravo, svete 123", "Здраво, свете 123"),
        ("Већ ћирилица", "Већ ћирилица"),
    ],
)
def test_serbian(latin, expected):
    assert to_cyrillic(latin, "sr") == expected


@pytest.mark.parametrize(
    ("latin", "expected"),
    [
        ("shto zhivot", "што живот"),
        ("gjavol kjerka", "ѓавол ќерка"),
        ("ǵavol ḱerka", "ѓавол ќерка"),
        ("ǵavol", "ѓавол"),
        ("dzvezda dzhamija", "ѕвезда џамија"),
        ("Čovek, ljubov, njiva", "Човек, љубов, њива"),
        ("Веќе е кирилица", "Веќе е кирилица"),
    ],
)
def test_macedonian(latin, expected):
    assert to_cyrillic(latin, "mk") == expected


def test_other_language_rejected():
    with pytest.raises(KeyError):
        to_cyrillic("hola", "es")
