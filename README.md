# Classification de disciplines médicales (medRxiv)

Prédiction automatique de la discipline médicale d'un preprint **medRxiv** (16 classes) à partir de son abstract uniquement. L'évaluation repose sur le **F1 macro**, afin d'accorder le même poids aux disciplines rares et fréquentes.

## Exploration et nettoyage des données
* **Déséquilibre des classes** : entre 324 et 1 060 abstracts par discipline (rapport de 1 à 3).
* **Suppression des doublons (anti-fuite)** : dédoublonnage des versions successives d'un même preprint (DOI/texte identiques) pour passer de 10 958 à **9 298 abstracts** uniques, et retrait des 26 *Withdrawal statements*.
* **Nettoyage** : suppression des balises HTML, codes de mise en page medRxiv (`O_FIG`, `O_SCPLOW`...) et liens.
* **Jeu de test** : gestion par défaut des 33 % d'abstracts vides (643 / 1 940).

## Approches testées

### 1. Machine Learning classique (`01_ml_classique.py`)
* **Validation** : validation croisée stratifiée à 5 plis (`Pipeline` complet pour éviter toute fuite de données).
* **Prétraitement** : tokenisation, minuscules, suppression des mots vides et lemmatisation (`spaCy`).
* **Vectorisation** : TF-IDF avec n-grammes de mots (1-2).
* **Déséquilibre** : pondération des classes (`class_weight="balanced"`).
* **Modèles** : SVM linéaire, Régression logistique, Naive Bayes, Random Forest, Gradient Boosting et Super-learner (stacking SVM + Naive Bayes).

### 2. Deep Learning avec PubMedBERT (`02_pubmedbert.py`)
* **Modèle** : `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`.
* **Troncature intelligente** : limitation à 256 tokens (64 premiers tokens pour le contexte + 190 derniers pour les résultats/conclusion).
* **Fine-tuning** : 3 époques, optimiseur AdamW ($lr = 2 \times 10^{-5}$) avec fonction de perte pondérée.
* **Ensemble** : combinaison des scores PubMedBERT ajusté + SVM linéaire ($w = 0,5$).

## Résultats (sur le même jeu de test interne)

| Approche | F1 macro | Exactitude |
| :--- | :---: | :---: |
| TF-IDF (1-2) + SVM linéaire | 0,569 | 0,586 |
| PubMedBERT gelé + Régression logistique | 0,548 | 0,562 |
| **PubMedBERT ajusté (*fine-tuning*)** | **0,613** | **0,624** |
| Combinaison PubMedBERT + SVM ($w=0,5$) | 0,612 | 0,623 |

Le *fine-tuning* de **PubMedBERT** apporte un gain de **+4,4 points de F1 macro** par rapport au meilleur modèle classique.

## Technologies
* **Python**, **scikit-learn**, **spaCy**
* **PyTorch**, **Hugging Face Transformers** (PubMedBERT)
