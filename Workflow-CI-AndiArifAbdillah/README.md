# Workflow-CI-AndiArifAbdillah

## Deskripsi
Repository untuk Kriteria 3: CI/CD Workflow dengan MLflow Project

**Author**: Andi Arif Abdillah

## Struktur Folder
```
Workflow-CI-AndiArifAbdillah/
├── MLProject/
│   ├── modelling.py               # Script training model
│   ├── MLproject                  # MLflow project file
│   ├── conda.yaml                 # Environment configuration
│   └── *.csv                      # Preprocessed datasets
└── .github/
    └── workflows/
        └── ci-*.yml               # GitHub Actions workflow
```

## Cara Menjalankan

### 1. Lokal
```bash
cd MLProject
mlflow run . --no-conda
```

### 2. CI/CD
Push ke GitHub, maka workflow akan otomatis menjalankan training.

## Level
- **Basic**: Training otomatis via CI
- **Skilled**: Training + upload artifacts
- **Advanced**: Training + Docker build & push

## MLflow Tracking
Model dan metrics akan di-track menggunakan MLflow.
