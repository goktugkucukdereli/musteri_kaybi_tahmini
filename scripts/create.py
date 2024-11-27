import os

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "scripts",
    "dashboards",
    "reports/figures"
]

files = [
    "README.md",
    "requirements.txt",
    "main.py",
    "reports/summary.md"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    open(file, 'w').close()

print("Proje dosya yapısı oluşturuldu!")
