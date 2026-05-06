# Prochaines etapes

## 1. Fixer des objectifs simples

On doit choisir 3 objectifs clairs:
- vendre plus de produits (cross-sell)
- faire moins d'erreurs de recommandation
- aider les agents a aller plus vite

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

## 4. Mieux verifier les explications

Chaque explication doit etre:
- vraie
- simple
- utile pour un agent

Il faut aussi garder des exemples bons et mauvais.

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

## Regle simple pour la suite

A chaque nouvelle fonction, il faut:
- un objectif clair
- un chiffre pour mesurer le succes
- un test automatique
- une preuve dans le dashboard
