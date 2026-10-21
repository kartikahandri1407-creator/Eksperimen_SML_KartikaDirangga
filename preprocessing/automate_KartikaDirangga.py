import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    """Preprocessing otomatis dataset Heart Failure.

    Langkahnya sama dengan notebook eksperimen:
    1. Hapus duplikat
    2. Ganti nilai 0 yang tidak valid (Cholesterol, RestingBP) dengan median
    3. Encoding kolom kategorikal
    4. Standarisasi kolom numerik
    5. Simpan hasil ke CSV
    """
    df = pd.read_csv(input_path)

    # 1. Hapus duplikat
    df = df.drop_duplicates()

    # 2. Ganti nilai 0 dengan median dari nilai valid
    for col in ['Cholesterol', 'RestingBP']:
        median_valid = df.loc[df[col] > 0, col].median()
        df[col] = df[col].replace(0, median_valid)

    # 3. Encoding kolom kategorikal
    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # 4. Standarisasi kolom numerik
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    hasil = preprocess_data(
        input_path='heart_raw/heart.csv',
        output_path='preprocessing/heart_preprocessing/heart_preprocessing.csv',
    )
    print("Preprocessing selesai:", hasil.shape)