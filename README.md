# 02 – IoT & Intelligence artificielle 

Mini-expérimentation de reconnaissance de mouvements.

## Installation

Prérequis : Python 3.11. (TensorFlow 2.21 ne fonctionne qu'avec Python 3.10 à 3.13, pas avec 3.14)

```bash
# créer et activer l'environnement virtuel
python3.11 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# installer les dépendances
pip install -r requirements.txt

# télécharger le dataset (~60 Mo, extrait dans data/)
python scripts/download_data.py
```

## Dataset

J'utilise le dataset **UCI HAR** (Human Activity Recognition Using Smartphones).

- 30 personnes avec un smartphone à la ceinture (Samsung Galaxy S II)
- accéléromètre + gyroscope 3 axes, échantillonnés à 50 Hz
- 6 activités : `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
- les signaux sont déjà découpés en fenêtres de 2,56 s (128 mesures) avec 50 % de chevauchement
- séparation train / test déjà faite **par personne** (21 personnes en train, 9 en test) : 7352 fenêtres de train, 2947 de test

C'est un dataset très utilisé, il a plus de 3 classes et il est déjà propre.
Le dataset n'est pas versionné dans le repo, il faut le télécharger avec le script ci-dessus.

## Sources

- UCI HAR Dataset : https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
