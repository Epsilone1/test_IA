"""
creer_icone.py
--------------
Dessine l'icone du projet (un petit reseau de neurones : 3 couches reliees)
et l'enregistre en test_IA.ico (16 a 256 px) a la racine du depot.

Lancement :  python creer_icone.py      (Pillow est installe avec torchvision)
"""

from pathlib import Path

from PIL import Image, ImageDraw

DOSSIER = Path(__file__).parent
FICHIER_ICO = DOSSIER / "test_IA.ico"
TAILLES = (16, 24, 32, 48, 64, 128, 256)

FOND_HAUT, FOND_BAS = (49, 27, 146), (13, 71, 161)   # violet -> bleu
LIEN = (128, 222, 234, 150)
NEURONE = (255, 255, 255, 255)
SORTIE = (255, 213, 79, 255)                          # neurone « gagnant » en jaune

# couches : abscisse, ordonnees des neurones (en fraction de l'image)
COUCHES = [
    (0.24, (0.30, 0.50, 0.70)),
    (0.50, (0.22, 0.41, 0.59, 0.78)),
    (0.76, (0.38, 0.62)),
]


def dessiner(taille=256):
    k = 4                                  # sur-echantillonnage : bords lisses
    s = taille * k
    fond = Image.new("RGBA", (s, s))
    d = ImageDraw.Draw(fond)
    for y in range(s):                     # degrade vertical
        t = y / (s - 1)
        d.line([(0, y), (s, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(FOND_HAUT, FOND_BAS)))
    masque = Image.new("L", (s, s), 0)
    ImageDraw.Draw(masque).rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=255)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    img.paste(fond, (0, 0), masque)

    liens = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    dl = ImageDraw.Draw(liens)
    for (x0, ys0), (x1, ys1) in zip(COUCHES, COUCHES[1:]):
        for y0 in ys0:
            for y1 in ys1:
                dl.line([(x0 * s, y0 * s), (x1 * s, y1 * s)], fill=LIEN, width=int(0.022 * s))
    img = Image.alpha_composite(img, liens)

    d = ImageDraw.Draw(img)
    r = 0.065 * s
    for i, (x, ys) in enumerate(COUCHES):
        for j, y in enumerate(ys):
            couleur = SORTIE if i == len(COUCHES) - 1 and j == 0 else NEURONE
            d.ellipse([x * s - r, y * s - r, x * s + r, y * s + r], fill=couleur)
    return img.resize((taille, taille), Image.LANCZOS)


if __name__ == "__main__":
    dessiner(256).save(FICHIER_ICO, format="ICO", sizes=[(t, t) for t in TAILLES])
    print("Icone creee :", FICHIER_ICO)
