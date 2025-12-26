# Eksperimen_SML_AndiArifAbdillah

## Deskripsi
Repository untuk Kriteria 1: Eksperimen dan Preprocessing Data Machine Learning

**Author**: Andi Arif Abdillah

## Struktur Folder
```
Eksperimen_SML_AndiArifAbdillah/
├── preprocessing/
│   ├── Eksperimen_AndiArifAbdillah.ipynb    # Notebook eksperimen
│   ├── automate_AndiArifAbdillah.py         # Script preprocessing otomatis
│   └── iris_preprocessing/                   # Output preprocessing
└── .github/
    └── workflows/
        └── preprocessing.yml                 # GitHub Actions workflow
```

## Cara Menjalankan

### 1. Eksperimen Manual (Basic)
```bash
cd preprocessing
jupyter notebook Eksperimen_AndiArifAbdillah.ipynb
```

### 2. Preprocessing Otomatis (Skilled)
```bash
cd preprocessing
python automate_AndiArifAbdillah.py
```

### 3. Workflow Otomatis (Advanced)
Push ke GitHub, maka GitHub Actions akan otomatis menjalankan preprocessing.

## Output
- `iris_raw.csv`: Dataset mentah
- `iris_train_preprocessed.csv`: Data training yang sudah diproses
- `iris_test_preprocessed.csv`: Data testing yang sudah diproses

## Dataset
Menggunakan Iris dataset (built-in di scikit-learn)
