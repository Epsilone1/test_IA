"""
apprentissage.py
----------------
Script d'ENTRAINEMENT.

Role : montrer des milliers d'exemples au modele, corriger ses erreurs,
puis sauvegarder ce qu'il a appris dans un fichier .pth.

Ensuite, execution.py rechargera ce .pth pour s'en servir sans reapprendre.

Lancement :  python apprentissage.py
"""

# Version_Control() doit etre appele AVANT tout import de torch :
# il installe / verifie la bonne version de PyTorch et peut relancer le script.
from Fonction.Versionning_Control import Version_Control
Version_Control()

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Fonction.Modele import Reseau


# --- Reglages ---
NB_EPOCHS = 5        # nombre de passages complets sur toutes les images
TAILLE_LOT = 64      # nombre d'images traitees en meme temps

DOSSIER = Path(__file__).parent
FICHIER_POIDS = DOSSIER / "Modele" / "modele.pth"


# --- Materiel : carte graphique si disponible, sinon processeur ---
appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Calcul sur", appareil)


# --- Donnees ---
# MNIST : 70 000 images de chiffres manuscrits, telechargees automatiquement.
# ToTensor transforme l'image en nombres entre 0 (noir) et 1 (blanc).
transformation = transforms.ToTensor()

# jeu d'entrainement : le modele apprend dessus
jeu_entrainement = datasets.MNIST(DOSSIER / "Donnees", train=True,
                                  download=True, transform=transformation)
# jeu de test : jamais vu pendant l'apprentissage, sert a verifier
# qu'il a vraiment compris et pas juste memorise
jeu_test = datasets.MNIST(DOSSIER / "Donnees", train=False,
                          download=True, transform=transformation)

# Le DataLoader distribue les images par lots et les melange
chargeur_entrainement = DataLoader(jeu_entrainement, batch_size=TAILLE_LOT, shuffle=True)
chargeur_test = DataLoader(jeu_test, batch_size=TAILLE_LOT)


# --- Modele ---
modele = Reseau().to(appareil)

# La fonction de perte mesure l'ecart entre la prediction et la bonne reponse.
fonction_perte = nn.CrossEntropyLoss()
# L'optimiseur decide comment corriger les poids a partir de cette erreur.
optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)


# --- Apprentissage ---
for epoch in range(1, NB_EPOCHS + 1):
    for images, etiquettes in chargeur_entrainement:
        images = images.to(appareil)
        etiquettes = etiquettes.to(appareil)

        # Les 5 etapes de l'apprentissage
        optimiseur.zero_grad()                            # 1. remet les corrections precedentes a zero
        predictions = modele(images)                      # 2. le modele devine
        perte = fonction_perte(predictions, etiquettes)   # 3. on mesure l'erreur
        perte.backward()                                  # 4. on calcule qui est responsable de l'erreur
        optimiseur.step()                                 # 5. on ajuste les poids du modele

    print(f"epoch {epoch}/{NB_EPOCHS} | perte {perte.item():.4f}")


# --- Verification sur les images jamais vues ---
bien_classees = 0
with torch.no_grad():   # pas de correction ici : on ne fait que tester
    for images, etiquettes in chargeur_test:
        images = images.to(appareil)
        etiquettes = etiquettes.to(appareil)
        predictions = modele(images)
        # argmax = le chiffre qui a obtenu le meilleur score
        bien_classees += (predictions.argmax(dim=1) == etiquettes).sum().item()

print(f"precision sur le test : {bien_classees / len(jeu_test) * 100:.2f}%")


# --- Sauvegarde de ce qui a ete appris ---
FICHIER_POIDS.parent.mkdir(exist_ok=True)
torch.save(modele.state_dict(), FICHIER_POIDS)
print("poids sauvegardes :", FICHIER_POIDS)