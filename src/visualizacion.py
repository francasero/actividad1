import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def generar_visualizaciones():
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)

    sns.set(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (8, 5)

    df_clientes = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    df_resenas = pd.read_csv("data/raw/datos_resenas.csv", sep=";")

    plt.figure()
    sns.countplot(x="sentimiento", data=df_resenas, order=["Negativo", "Neutral", "Positivo"])
    plt.title("Distribucion de la variable objetivo (sentimiento)")
    plt.xlabel("Sentimiento expresado en la resena")
    plt.ylabel("Cantidad de resenas")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribucion_sentimiento.png"))
    plt.close()

    df_resenas["longitud_texto"] = df_resenas["texto_resena"].astype(str).str.len()
    plt.figure()
    sns.boxplot(x="sentimiento", y="longitud_texto", data=df_resenas, order=["Negativo", "Neutral", "Positivo"])
    plt.title("Longitud del texto de la resena segun sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad de caracteres")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "longitud_texto_sentimiento.png"))
    plt.close()
    df_resenas.drop(columns=["longitud_texto"], inplace=True)

    cat_clientes = df_clientes.select_dtypes(include=["object", "string"]).columns
    for col in cat_clientes:
        plt.figure(figsize=(8, 4))
        order = df_clientes[col].value_counts().index
        sns.countplot(y=col, data=df_clientes, order=order)
        plt.title(f"Distribucion de {col} (Clientes)")
        plt.xlabel("Cantidad")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"clientes_{col}.png"))
        plt.close()

    cat_resenas = df_resenas.select_dtypes(include=["object", "string"]).columns.drop(
        ["texto_resena", "fecha_resena", "sentimiento"]
    )
    for col in cat_resenas:
        plt.figure(figsize=(8, 4))
        order = df_resenas[col].value_counts().index
        sns.countplot(y=col, data=df_resenas, order=order)
        plt.title(f"Distribucion de {col} (Resenas)")
        plt.xlabel("Cantidad")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"resenas_{col}.png"))
        plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_clientes.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Clientes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlacion_clientes.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_resenas.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlaciones - Resenas")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlacion_resenas.png"))
    plt.close()

    print(f"Graficos exportados correctamente en {output_dir}")


if __name__ == "__main__":
    generar_visualizaciones()