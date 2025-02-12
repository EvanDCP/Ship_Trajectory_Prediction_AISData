# Prédiction de Trajectoire des Navires

## Introduction

Ce projet vise à prédire la trajectoire des navires dans une zone géographique spécifique, ici la Méditerranée, en exploitant les données AIS (Automatic Identification System). Il repose sur des techniques avancées de machine learning et de deep learning, notamment l'utilisation du modèle GPT-2, pour analyser les schémas de déplacement et anticiper les trajectoires futures des navires. Le processus englobe plusieurs étapes essentielles, allant du prétraitement des données à l'entraînement du modèle, en passant par l'analyse exploratoire et la visualisation interactive des résultats. 

## Installation des prérequis

Avant d'exécuter ce projet, il est impératif d'installer plusieurs bibliothèques nécessaires à la manipulation des données, à la visualisation et à l'entraînement du modèle. Les bibliothèques incluent notamment `folium`, `geopandas`, `transformers`, `seaborn`, `scikit-learn`, `haversine` et `torch`. Ces dépendances peuvent être installées via la commande suivante :

```bash
pip install folium geopandas tokenizers datasets transformers accelerate psycopg seaborn scikit-learn haversine torch matplotlib
```

## Chargement et Prétraitement des Données

Les données AIS sont chargées à partir d’un fichier parquet, contenant des informations précises sur la position des navires, leur vitesse et leur cap. Un prétraitement approfondi est effectué pour garantir la qualité des données utilisées dans le modèle. Ce processus comprend :

1. **Filtrage des valeurs aberrantes** : Suppression des vitesses excessives ou négatives, ainsi que des angles incohérents.
2. **Nettoyage des entrées invalides** : Retrait des navires ayant un nombre insuffisant de signaux pour un suivi cohérent.
3. **Structuration des trajectoires** : Transformation des données brutes en séquences exploitables par un modèle de langage naturel.

## Analyse Exploratoire et Visualisation

Une analyse approfondie des données est réalisée à l’aide des bibliothèques `matplotlib` et `seaborn`. Cette étape permet d’identifier les tendances sous-jacentes et de détecter d’éventuelles anomalies. Parmi les outils d’analyse utilisés, on retrouve :

- **Histogrammes des vitesses et des caps** pour comprendre la répartition des valeurs.
- **Clusters de trajectoires avec KMeans** pour segmenter les déplacements en catégories cohérentes.
- **Cartes interactives Folium** afin de visualiser les itinéraires des navires et les prédictions du modèle.

## Préparation des Données pour le Modèle

Afin d’entraîner un modèle basé sur GPT-2, les trajectoires des navires sont converties en texte structuré. Cette transformation est essentielle pour permettre au modèle de traiter les séquences temporelles de manière efficace. Les étapes clés de cette transformation incluent :

- **Tokenisation des trajectoires** avec un `tokenizer GPT-2` afin de segmenter les données en unités exploitables.
- **Découpage des données** en ensembles d'entraînement et de validation pour optimiser l'apprentissage.
- **Normalisation des séquences** afin d’harmoniser les formats et d’éviter les biais d’entraînement.

## Entraînement du Modèle GPT-2

L’entraînement du modèle repose sur la bibliothèque `Hugging Face Transformers`, qui permet d’adapter le modèle GPT-2 à la prédiction des trajectoires maritimes. L’entraînement est configuré avec les paramètres suivants :

- **Nombre d’époques** : 20
- **Taux d’apprentissage** : 5e-4 avec une planification de type cosinus
- **Batch size** : 16 pour optimiser la gestion de la mémoire GPU
- **Stratégie de validation** : évaluation du modèle à intervalles réguliers pour éviter le sur-ajustement

Le modèle est entraîné sur un GPU si disponible, ce qui accélère considérablement le processus d’apprentissage.

## Sauvegarde et Chargement du Modèle

Une fois l’entraînement terminé, le modèle est sauvegardé afin d’être réutilisé sans nécessiter un nouvel apprentissage. Une fonction dédiée permet de charger rapidement le modèle et son tokenizer pour générer de nouvelles prédictions.

```python
def load_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer
```

## Génération et Évaluation des Prédictions

Les trajectoires prédites sont ensuite comparées aux trajectoires réelles afin d’évaluer la performance du modèle. Cette évaluation repose sur plusieurs métriques, notamment :

- **Distance de Haversine** : Calcul de la distance entre les points prévus et les points réels.
- **Visualisation sur carte interactive** : Comparaison graphique des trajectoires réelles et prédites.
- **Score de précision** : Mesure basée sur la proximité des points prédits par rapport aux positions réelles des navires.

L'affichage interactif des trajectoires à l’aide de `Folium` permet d’avoir une vue détaillée des performances du modèle en fonction des différents scénarios testés.

## Utilisation du Projet

Pour exécuter le projet, il suffit de lancer le script principal en ligne de commande :

```bash
python Trajectory_Prediction_sog.py
```

Les résultats des analyses et des prédictions seront affichés directement. Il est possible de tester le modèle sur de nouvelles données en modifiant le fichier d'entrée. Cette flexibilité permet d’évaluer la capacité du modèle à s’adapter à différents contextes de navigation.

## Améliorations Possibles

Plusieurs axes d'amélioration peuvent être explorés pour optimiser les performances du modèle :

- **Utilisation de modèles pré-entraînés plus performants** comme GPT-3 ou des architectures spécialisées pour les séries temporelles.
- **Augmentation des données d’entraînement** en intégrant des bases de données AIS plus vastes et diversifiées.
- **Application du transfert d’apprentissage** afin d'affiner les performances sur des scénarios spécifiques.
- **Optimisation des hyperparamètres** pour ajuster le modèle aux particularités des données maritimes.

## Conclusion

Ce projet démontre comment l’intelligence artificielle et le deep learning peuvent être appliqués à la prédiction des trajectoires maritimes. Il illustre l’importance du prétraitement des données et de l’analyse exploratoire pour améliorer la performance des modèles. En combinant les techniques de traitement du langage naturel et du machine learning, cette approche offre une solution efficace et flexible pour anticiper les mouvements des navires sur une zone donnée. Grâce aux améliorations futures et à l’intégration de nouvelles données, ce projet pourra être étendu pour fournir des prévisions toujours plus précises et robustes.
