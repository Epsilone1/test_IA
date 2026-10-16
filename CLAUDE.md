# test_IA

Essais d'apprentissage automatique avec PyTorch (scripts console, pas d'interface graphique).
Commentaires en français.

**L'utilisateur (Vincent)** veut un résumé en début de réponse et pas de flatterie.

## Arborescence
```
IA_projet/
  apprentissage.py            entraîne le réseau sur MNIST, sauvegarde Modele/modele.pth
  execution.py                recharge modele.pth et mesure la réussite sur le jeu de test
  Fonction/Modele.py          architecture du réseau (Reseau : 784 -> 2048 -> 512 -> 10)
  Fonction/Versionning_Control.py  installe / vérifie PyTorch (CPU ou GPU cu130) ; à appeler
                              AVANT tout import de torch (peut relancer le script)
  Fonction/Temp.json          état de Versionning_Control (version installée, mode CPU/GPU)
  Donnees/MNIST/              jeu de données (téléchargé automatiquement)
  Modele/modele.pth           poids appris
Fichier_persistant/test.py    essai de variable persistante dans un JSON
IA_projet.rar                 archive du projet
creer_icone.py                dessine test_IA.ico
test_IA.ico                   icône du projet
```

Lancer : `cd IA_projet` puis `python apprentissage.py` ou `python execution.py`
(les imports `from Fonction...` supposent que le dossier courant du script est `IA_projet/`).

## Icône
`test_IA.ico` à la racine (réseau de neurones, 16 à 256 px), régénérable par
`python creer_icone.py`. Programmes console : l'icône n'apparaît que sur un exe compilé
(fichier, raccourci, fenêtre console, barre des tâches) ; le Compilateur EXE la trouve en
remontant du dossier du script jusqu'à la racine du dépôt.
