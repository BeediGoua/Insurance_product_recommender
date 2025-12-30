# Analyse du Projet - Recommandeur de Produits d'Assurance

Ce document synthétise l'état actuel du projet et les performances obtenues.

## 1. État d'Avancement du Projet

Le projet est bien structuré et dispose d'une base solide pour la recommandation de produits.

### Composants Implémentés
- **Modèle Baseline** : Un modèle de recommandation conditionnelle (`conditional_baseline`) a été mis en œuvre. Il se base sur les probabilités d'association entre produits.
- **Pipeline de Données** : Des scripts de nettoyage (`cleaning.py`) et de préparation sont fonctionnels.
- **Cadre d'Évaluation** : Un système robuste d'évaluation par "masking" (`evaluate_V2.py`) est en place, permettant de simuler des recommandations réelles en masquant des produits détenus.
- **Analyses Additionnelles** : Des modules pour la segmentation client, l'analyse de produits et la détection de "drift" (dérive des données) sont présents dans `src/analytics`.
- **Interface Utilisateur** : Une application Streamlit a été amorcée (`app/Accueil.py`) pour présenter les résultats et les recommandations.
- **Documentation et Recherche** : Plusieurs notebooks (EDA, Baseline, Audit) documentent l'exploration initiale et les premiers résultats.

### Travaux en Cours (Induits)
- Un fichier `nouveau_model.md` a été détecté dans les notes, suggérant la conception d'un modèle plus avancé (potentiellement à base de gradient boosting ou de deep learning) pour dépasser la baseline.

---

## 2. Performances (Baseline V0)

Les scores actuels de la baseline `conditional_baseline` sont très encourageants :

| Métrique | Score |
| :--- | :--- |
| **Hit@1** | **82.07%** |
| **Hit@3** | **91.95%** |
| **Hit@5** | **96.10%** |
| **Hit@10** | **98.65%** |
| **MRR** (Mean Reciprocal Rank) | **87.81%** |

> [!NOTE]
> Le score **Hit@1** de ~82% indique que dans 8 cas sur 10, le produit masqué est le premier recommandé par le système.
> Le **MRR** de ~87% confirme la forte pertinence du classement global.

---

## 3. Structure du Code

- `src/baselines/` : Logique du modèle actuel.
- `src/evaluation/` : Outils de mesure de performance.
- `src/analytics/` : Insights métier (segmentation, drift).
- `app/` : Interface de démonstration.
- `artifacts/` : Stockage des versions de modèles et des métriques associées.

---

## 4. Recommandations pour la suite
1. **Implémenter le "Nouveau Modèle"** : Utiliser les caractéristiques socio-démographiques pour affiner les recommandations au-delà des simples associations de produits.
2. **Enrichir l'Application Streamlit** : Ajouter une page de simulation pour permettre aux équipes CRM de tester des profils clients spécifiques.
3. **Validation Croisée** : Utiliser les scripts d'analyse de drift pour s'assurer de la robustesse temporelle du modèle.
