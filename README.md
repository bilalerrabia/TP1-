
# Modèle de Régression Logistique avec Perceptron

Ce projet implémente un modèle de régression logistique pour la classification binaire en utilisant un perceptron. Le modèle est construit à partir de fonctions essentielles comme l'initialisation des paramètres, le calcul des prédictions, la fonction de coût, les gradients, et la mise à jour des paramètres. Ensuite, un modèle complet est créé avec les fonctions pour l'entraînement et la prédiction.

## Description

Le programme exécute les étapes suivantes :

1. **Initialisation des paramètres** :
   Les poids (W) et le biais (b) sont initialisés de manière aléatoire.

2. **Modèle de prédiction** :
   Un modèle linéaire z = W^T X + b est utilisé, suivi d'une fonction sigmoïde pour obtenir la probabilité d'appartenance à la classe positive.

3. **Calcul du coût** :
   La fonction de coût utilisée est la log-loss, qui mesure la différence entre les prédictions et les étiquettes réelles.

4. **Calcul des gradients** :
   Les gradients sont calculés pour les poids et le biais afin de minimiser la fonction de coût pendant l'entraînement.

5. **Mise à jour des paramètres** :
   Les poids et le biais sont mis à jour à l'aide de la descente de gradient avec un taux d'apprentissage (α) spécifié.

6. **Entraînement du modèle** :
   Le modèle est entraîné pour un nombre d'itérations donné, et le coût est calculé et affiché à chaque itération.

7. **Prédiction et évaluation** :
   Après l'entraînement, le modèle prédit les classes des nouveaux exemples et calcule l'accuracy (précision) en comparant les prédictions aux étiquettes réelles.

8. **Visualisation** :
   - Les données d'entraînement et les nouvelles données sont affichées dans un graphique.
   - La frontière de décision est tracée pour montrer la séparation entre les classes.

## Structure du Code

- **Fonction `initialisation(X)`** : Initialise les poids W et le biais b de manière aléatoire.
- **Fonction `model(X, W, b)`** : Calcule la sortie du modèle en appliquant la fonction sigmoïde à la combinaison linéaire des entrées et des paramètres.
- **Fonction `log_loss(A, y)`** : Calcule la fonction de coût, qui est la log-loss.
- **Fonction `gradients(A, X, y)`** : Calcule les gradients des poids et du biais.
- **Fonction `update(W, b, dW, db, alpha)`** : Met à jour les poids et le biais en utilisant les gradients et un taux d'apprentissage.
- **Fonction `predict(X, W, b)`** : Prédit la classe des exemples en renvoyant 1 si la probabilité est supérieure ou égale à 0.5, sinon 0.
- **Classe `ArtificialNeurons`** : Combine toutes les fonctions ci-dessus pour entraîner un modèle de neurone artificiel avec une descente de gradient et prédire les résultats.

## Installation

1. Clone ce repository ou télécharge le fichier Python contenant le code.
2. Assure-toi d'avoir Python 3.x installé ainsi que les bibliothèques suivantes :
   - numpy
   - matplotlib
   - scikit-learn

Tu peux installer les dépendances nécessaires avec pip :
```bash
pip install numpy matplotlib scikit-learn
```

## Exécution

1. Exécute le script Python après avoir défini les données d'entrée et les paramètres comme le taux d'apprentissage (α) et le nombre d'itérations (n_iter).
2. Les erreurs, la précision (accuracy) et la frontière de décision seront affichées sur un graphique.
3. Tu peux tester la prédiction du modèle sur de nouveaux exemples de données et visualiser la performance du modèle.

## Résultats attendus

- Le coût (log-loss) au fil des itérations.
- La précision du modèle (accuracy) sur l'ensemble d'entraînement.
- Un graphique montrant la séparation des classes et les nouvelles données ajoutées.
- Une frontière de décision tracée entre les deux classes.

