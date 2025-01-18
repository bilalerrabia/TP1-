### README: Classification avec un Arbre de Décision - "Jouer au Tennis"

---

#### **Aperçu du Projet**
Ce projet utilise un **arbre de décision** pour prédire si une partie de tennis sera jouée ou non en fonction de facteurs météorologiques comme les prévisions, la température, l'humidité et le vent. Le modèle est construit avec la bibliothèque **Scikit-learn** et inclut une évaluation complète des performances du modèle.

---

#### **Caractéristiques**
- Construction et entraînement d’un arbre de décision basé sur le critère **entropie**.
- Encodage des données catégoriques pour l’utilisation dans un modèle de machine learning.
- Évaluation du modèle avec :
  - **Précision (Accuracy)**,
  - **Rappel (Recall)**,
  - **F1-score**.
- Visualisation de :
  - L’arbre de décision,
  - La matrice de confusion.
- Analyse des performances en fonction de la profondeur maximale de l'arbre.

---

#### **Données**
Les données utilisées sont un ensemble fictif représentant les conditions météorologiques et leur influence sur la décision de jouer ou non au tennis.

- **Colonnes** :
  - `Prévisions`: Ensoleillé, Nuageux, Pluie
  - `Température`: Chaud, Moyen, Frais
  - `Humidité`: Haute, Normale
  - `Vent`: Oui, Non
  - `Jouer`: Oui (jouer) ou Non (ne pas jouer)

Exemple de données :
| Prévisions  | Température | Humidité | Vent | Jouer |
|-------------|-------------|----------|------|-------|
| Ensoleillé  | Chaud       | Haute    | Non  | Non   |
| Pluie       | Frais       | Normale  | Oui  | Non   |

---

#### **Prérequis**
- **Python 3.7+**
- Bibliothèques Python :
  - `scikit-learn`
  - `pandas`
  - `matplotlib`

Installation des bibliothèques nécessaires :
```bash
pip install scikit-learn pandas matplotlib
```

---

#### **Instructions**
1. **Exécuter le script** :
   Lancez le fichier Python pour exécuter le programme :
   ```bash
   python <nom_du_fichier>.py
   ```

2. **Étapes du programme** :
   - **Encodage des données** : Transformation des variables catégoriques en données numériques à l'aide de `pd.get_dummies`.
   - **Construction du modèle** : Entraînement d’un arbre de décision avec un ensemble d’entraînement.
   - **Évaluation du modèle** :
     - Affichage des métriques comme la précision, le rappel et le F1-score.
     - Visualisation de l’arbre de décision et de la matrice de confusion.
   - **Analyse de la profondeur de l'arbre** : Étude des performances du modèle pour différentes profondeurs maximales.

3. **Résultats** :
   - **Arbre de décision** : Une représentation visuelle de l’arbre est affichée.
   - **Matrice de confusion** : Permet de comprendre les erreurs du modèle.
   - **Performance globale** : Les métriques sont affichées dans la console.

---

#### **Fonctionnalités clés**
- **Construction de l'arbre** :
  L’arbre de décision est construit à l’aide de `DecisionTreeClassifier` avec le critère `entropy`.
  
- **Visualisation** :
  - Affichage de l’arbre avec `plot_tree`.
  - Matrice de confusion avec `ConfusionMatrixDisplay`.

- **Métriques** :
  - **Précision (Accuracy)** : Mesure de la proportion des prédictions correctes.
  - **Rappel (Recall)** : Proportion des cas positifs correctement identifiés.
  - **F1-score** : Moyenne harmonique entre précision et rappel.

---

#### **Personnalisation**
- Vous pouvez modifier la profondeur maximale de l’arbre pour tester ses performances (section de code `for depth in range(1, 6)`).
- Remplacez ou ajoutez des données dans l'ensemble pour tester le modèle avec différents scénarios.

---

#### **Exemple de Résultats**
1. **Précision sur l'ensemble de test** :
   ```
   Précision sur l'ensemble de test : 100.00%
   ```

2. **Arbre de décision** (exemple en texte) :
   ```
   |--- Humidité_Normale <= 0.50
   |   |--- Température_Frais <= 0.50
   |   |   |--- classe: Non
   |--- classe: Oui
   ```

3. **Métriques** :
   ```
   Taux de reconnaissance (accuracy) : 100.00%
   Rappel (Recall) : 1.00
   F1-Score : 1.00
   ```

---

#### **Améliorations possibles**
- Ajouter d'autres algorithmes de classification pour comparer les performances (ex. : SVM, KNN).
- Implémenter une validation croisée pour une évaluation plus robuste.
- Étendre les données pour inclure des cas plus variés.

---

