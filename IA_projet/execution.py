"""
execution.py
------------
Script d'UTILISATION.

Role : recharger le modele deja entraine par apprentissage.py
et lui faire deviner le chiffre de plusieurs images du jeu de test,
puis afficher son pourcentage de reussite.

Ici on n'apprend plus rien : le modele ne change pas, il ne fait que repondre.

Lancement :  python execution.py
"""

# Version_Control() doit etre appele AVANT tout import de torch :
# il installe / verifie la bonne version de PyTorch et peut relancer le script.
from Fonction.Versionning_Control import Version_Control
Version_Control()

from pathlib import Path

import torch
from torchvision import datasets, transforms

from Fonction.Modele import Reseau


# --- Reglages ---
NB_IMAGES = 10000      # nombre d'images a tester (10 000 disponibles au maximum)

DOSSIER = Path(__file__).parent
FICHIER_POIDS = DOSSIER / "Modele" / "modele.pth"


# --- Materiel : carte graphique si disponible, sinon processeur ---
appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Reconstruction du modele et rechargement des poids appris ---
# On cree la meme structure vide qu'a l'entrainement...
modele = Reseau().to(appareil)
# ...puis on y remet les valeurs sauvegardees dans le .pth
modele.load_state_dict(torch.load(FICHIER_POIDS, map_location=appareil))
modele.eval()   # mode utilisation : le modele ne s'entraine plus


# --- Images a analyser ---
jeu_test = datasets.MNIST(DOSSIER / "Donnees", train=False,
                          download=True, transform=transforms.ToTensor())


# --- Boucle de prediction ---
bien_devinees = 0

for numero in range(NB_IMAGES):
    image, bonne_reponse = jeu_test[numero]

    # Le modele attend toujours un LOT : on en fabrique un d'une seule image
    image = image.unsqueeze(0).to(appareil)

    with torch.no_grad():   # pas de correction ici : on ne fait que lire la reponse
        scores = modele(image)

    # argmax = le chiffre qui a obtenu le meilleur score
    chiffre = scores.argmax(dim=1).item()

    if chiffre == bonne_reponse:
        bien_devinees += 1


# --- Resultat ---
print(f"reussite : {bien_devinees}/{NB_IMAGES} = {bien_devinees / NB_IMAGES * 100:.1f}%")