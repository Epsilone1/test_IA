"""
execution.py
------------
Script d'UTILISATION.

Role : recharger le modele deja entraine par apprentissage.py
et lui faire deviner le chiffre present sur une image.

Ici on n'apprend plus rien : le modele ne change pas, il ne fait que repondre.

Lancement :  python execution.py                  -> prend une image du jeu de test
             python execution.py mon_image.png    -> prend ton image
"""

# Version_Control() doit etre appele AVANT tout import de torch :
# il installe / verifie la bonne version de PyTorch et peut relancer le script.
from Fonction.Versionning_Control import Version_Control
Version_Control()

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

from Fonction.Modele import Reseau


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


# --- Recuperation de l'image a analyser ---
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    # Une image fournie par toi
    image = Image.open(sys.argv[1]).convert("L")    # "L" = noir et blanc
    image = image.resize((28, 28))                  # meme taille qu'a l'entrainement
    # MNIST contient des chiffres blancs sur fond noir.
    # Si ton image est un chiffre noir sur fond blanc, decommente la ligne suivante :
    # from PIL import ImageOps; image = ImageOps.invert(image)
    tenseur = transforms.ToTensor()(image)
    bonne_reponse = None
else:
    # Sinon on prend la premiere image du jeu de test, dont on connait la reponse
    jeu_test = datasets.MNIST(DOSSIER / "Donnees", train=False,
                              download=True, transform=transforms.ToTensor())
    tenseur, bonne_reponse = jeu_test[0]

# Le modele attend toujours un LOT d'images : on en fabrique un d'une seule image
tenseur = tenseur.unsqueeze(0).to(appareil)


# --- Prediction ---
with torch.no_grad():   # pas de calcul de correction : on ne fait que lire la reponse
    scores = modele(tenseur)

# argmax = le chiffre qui a obtenu le meilleur score
chiffre = scores.argmax(dim=1).item()
# softmax transforme les scores bruts en pourcentages de confiance
confiance = torch.softmax(scores, dim=1)[0, chiffre].item()

print(f"chiffre predit : {chiffre}  (confiance {confiance * 100:.1f}%)")
if bonne_reponse is not None:
    print(f"bonne reponse  : {bonne_reponse}")