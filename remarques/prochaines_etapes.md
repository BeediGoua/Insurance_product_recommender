# Prochaines etapes

## 1. Fixer des objectifs simples

On garde 3 objectifs clairs:
- vendre plus de produits (cross-sell)
- faire moins d'erreurs de recommandation
- aider les agents a aller plus vite

### Etapes a faire

1. Choisir les 3 KPI.
2. Ecrire une formule simple pour chaque KPI.
3. Mesurer la baseline actuelle.
4. Fixer une cible a 1 mois et 3 mois.
5. Nommer un responsable par KPI.

### Template KPI (a remplir)

| KPI | Formule | Baseline actuelle | Cible 1 mois | Cible 3 mois | Responsable |
|---|---|---|---|---|---|
| Cross-sell | ventes issues reco / reco envoyees | ... | ... | ... | ... |
| Taux d'erreurs | reco invalides / reco totales | ... | ... | ... | ... |
| Temps agent | temps moyen par dossier | ... | ... | ... | ... |

### Regle de validation

Un KPI est valide seulement si:
- la formule est claire
- la source de donnees est connue
- la baseline est mesuree
- la cible est chiffree

## 2. Mieux evaluer le modele

Aujourd'hui, l'evaluation est trop faible.

Il faut:
- un vrai jeu de test avec beaucoup de clients
- des chiffres clairs: Hit@1, Hit@3, Hit@5, MRR
- des resultats par groupe de clients (age, branche, etc.)

## 3. Verifier les regles metier

Il faut verifier que:
- les produits interdits ne sortent jamais
- les produits deja possedes ne sortent jamais
- les regles ne bloquent pas trop de bons produits

### Etapes a faire

1. Lancer le moteur sans regles.
2. Lancer le moteur avec regles.
3. Comparer avant/apres.
4. Compter les violations et blocages.
5. Analyser les raisons de blocage les plus frequentes.

### Mesures a suivre

- Taux de violation = violations / recommandations
- Taux de blocage = produits bloques / produits proposes avant regles
- Sur-blocage = cas ou un bon produit est bloque

### Template de suivi regles (a remplir)

| Mesure | Valeur | Seuil cible | Statut |
|---|---|---|---|
| Taux de violation produits interdits | ... | 0% | ... |
| Taux de violation produits deja possedes | ... | 0% | ... |
| Taux de blocage global | ... | a definir | ... |
| Sur-blocage estime | ... | a reduire | ... |

### Tableau avant/apres regles (a remplir)

| Cas client | Top-k avant regles | Top-k apres regles | Produits bloques | Raison principale |
|---|---|---|---|---|
| C001 | ... | ... | ... | ... |
| C002 | ... | ... | ... | ... |
| C003 | ... | ... | ... | ... |

## 4. Mieux verifier les explications

Chaque explication doit etre:
- vraie
- simple
- utile pour un agent

Il faut aussi garder des exemples bons et mauvais.

### Etapes a faire

1. Prendre un echantillon de 50 a 100 cas.
2. Noter chaque explication sur 3 criteres: vraie, simple, utile.
3. Garder 10 bons exemples et 10 mauvais exemples.
4. Corriger les erreurs les plus frequentes.
5. Refaire la mesure apres correction.

### Grille de notation (a remplir)

| Cas | Vraie (0-5) | Simple (0-5) | Utile (0-5) | Commentaire |
|---|---|---|---|---|
| C001 | ... | ... | ... | ... |
| C002 | ... | ... | ... | ... |
| C003 | ... | ... | ... | ... |

### Score global explication

- Score vrai moyen = ... / 5
- Score simple moyen = ... / 5
- Score utile moyen = ... / 5
- Score global = (vrai + simple + utile) / 3

### Bibliotheque d'exemples

#### Bons exemples
- Exemple 1: ...
- Exemple 2: ...

#### Mauvais exemples
- Exemple 1: ...
- Exemple 2: ...

#### Erreurs frequentes a corriger
- Raison trop vague
- Phrase trop longue
- Explication non liee au profil client

## 5. Tester les agents IA

Si on garde les agents, il faut mesurer:
- s'ils appellent les bons outils
- leur temps de reponse
- leur cout

## 6. Renforcer les tests

Il faut ajouter:
- des tests de non-regression
- des tests entre modules (decisionflow -> policy -> explanation -> audit)
- une verification automatique dans la CI

## 7. Faire un test terrain

Apres l'offline, il faut tester en vrai:
- petit pilote
- comparaison avec la methode actuelle
- mesure des gains reels

## Ordre conseille

1. Refaire l'evaluation correctement.
2. Verifier les regles metier.
3. Tester en terrain reel.
4. Optimiser les agents IA ensuite.

## Mini planning (2 semaines)

### Semaine 1
- Jour 1-2: finaliser KPI et baseline (point 1)
- Jour 3-4: campagne de verification regles (point 3)
- Jour 5: revue resultats et priorites

### Semaine 2
- Jour 1-3: audit des explications + scoring (point 4)
- Jour 4: correction rapide des erreurs frequentes
- Jour 5: mesure finale et decision go/no-go pour pilote

## Regle simple pour la suite

A chaque nouvelle fonction, il faut:
- un objectif clair
- un chiffre pour mesurer le succes
- un test automatique
- une preuve dans le dashboard
