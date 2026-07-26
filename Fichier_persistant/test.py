import json
from pathlib import Path

Fichier = Path(__file__).parent / "persistant.json"

Variable = 0

while True:
    try:
        if Fichier.exists():
            data = json.loads(Fichier.read_text(encoding="utf-8"))
            Variable = data.get("variable", Variable)

    except json.JSONDecodeError:
        pass

    print(f"Variable: {Variable}")

    Variable = input("Entrez une nouvelle valeur pour la variable : ")

    Fichier.write_text(json.dumps({"variable": Variable}), encoding="utf-8")

    print(f"Nouvelle valeur de la variable enregistrée : {Variable}")
