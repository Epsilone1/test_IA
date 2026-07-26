"""
Modele.py
---------
Definition de l'architecture du reseau de neurones.

Ce fichier ne fait QUE decrire la forme du modele.
Il n'apprend rien et ne charge rien : c'est un plan de construction.
Il est importe par apprentissage.py (pour entrainer) et par execution.py
(pour reconstruire la meme structure avant de recharger les poids appris).
"""

import torch.nn as nn


class Reseau(nn.Module):
    """
    Entree : une image noir et blanc de 28x28 pixels
    Sortie : 10 nombres, un score par chiffre possible (0 a 9)
    """

    def __init__(self):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Flatten(),          # met les 28x28 pixels sur une seule ligne = 784 valeurs
            nn.Linear(784, 2048),   # 784 entrees -> 2048 neurones
            nn.ReLU(),             # garde les valeurs positives, met le reste a 0
            nn.Linear(2048, 512),   # 2048 entrees -> 512 neurones
            nn.ReLU(),             # garde les valeurs positives, met le reste a 0
            nn.Linear(512, 10),    # 512 neurones -> 10 scores de sortie
        )

    def forward(self, x):
        """
        Chemin suivi par les donnees dans le reseau.
        PyTorch appelle cette methode quand on ecrit modele(x).
        """
        return self.couches(x)