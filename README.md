# Recommandation d’assurance : proposer le bon produit au bon client

## 1) De quoi parle ce projet

Ce projet vise à construire un système de recommandation pour une assurance : pour chaque client, nous voulons identifier **quel produit a le plus de sens à proposer**, et produire un **Top-K** (ex. 1, 3 ou 5 produits) utilisable par une équipe CRM/marketing.

L’idée n’est pas de faire “un modèle de compétition”, mais un outil qui aide à prendre une décision concrète : **qui contacter** et **quoi proposer**.

---

## 2) Pourquoi ce projet existe (le besoin réel)

Dans une assurance, contacter tous les clients coûte cher et donne souvent de mauvais résultats.
En plus, proposer un produit que le client possède déjà est inutile et dégrade la relation.

Le besoin est donc simple :

* mieux cibler (contacter moins de personnes, mais les bonnes),
* mieux personnaliser (proposer le produit le plus pertinent),
* et **faire confiance au score** (si on dit 0,8, ça doit vraiment signifier “forte probabilité”).

---

## 3) Ce qu’on veut obtenir à la fin (résultat concret)

Pour chaque client, le système doit fournir :

1. un classement des produits recommandés (Top-K),
2. un score de confiance (probabilité) lisible,
3. une règle simple d’action (ex. “contacter seulement les clients les plus actionnables”).

---

## 4) Ce que les données permettent réellement

Les données décrivent :

* le profil du client (quelques informations socio-démographiques et des codes),
* les produits déjà détenus par le client (21 produits, indiqués par 0/1).

Le jeu est construit de façon particulière : dans les données de test, **un produit réellement détenu a été masqué**.
Le but est donc d’apprendre la logique suivante :

> “vu le profil du client et les autres produits qu’il possède, quel est le produit le plus probable qui manque ?”

C’est une situation proche d’un cas d’usage réel de recommandation : on apprend des **associations** entre produits et profils pour proposer le produit le plus pertinent.

---

## 5) En quoi ce projet est différent des projets classiques

Beaucoup de projets se limitent à “obtenir un bon score” sur une métrique et à s’arrêter là.

Ici, nous voulons aller plus loin :

* produire un Top-K utilisable (pas juste une prédiction),
* garantir qu’on ne recommande pas un produit déjà détenu,
* rendre les scores **fiables** pour que la recommandation serve vraiment à décider qui contacter,
* évaluer l’impact de la recommandation comme une vraie campagne (priorisation, budget, top-p%).

---

## 6) Limites (honnêtes) dès le départ

Ces données ne permettent pas de dire :

* “le prochain produit acheté dans le temps” (il n’y a pas d’historique chronologique d’achats).

Elles permettent de construire :

* un moteur de recommandation basé sur la **compatibilité** (produits qui vont ensemble) et le profil client,
  ce qui est déjà très utile en CRM.

---

## 7) À qui ce projet peut servir

* Équipe marketing / CRM : prioriser les clients et personnaliser les campagnes.
* Analyste data : comprendre les associations entre produits et segments clients.
* Produit / business : simuler une stratégie “contacter top-p%” avec des scores fiables.

---

Si tu veux, prochaine étape (toujours simple, sans code) : je propose une section “Comment évaluer si ça marche ?” expliquée en langage non-tech (Top-K, fiabilité du score, impact campagne), pour que n’importe qui comprenne comment on juge la réussite du projet.
