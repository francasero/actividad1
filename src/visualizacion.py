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

def generar_graficos_bivariados():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("reports/figures", exist_ok=True)
    df = pd.read_csv("data/raw/datos_resenas.csv", sep=";")

    orden = ["Negativo", "Neutral", "Positivo"]
    colores = ["#d9534f", "#f0ad4e", "#5cb85c"]

    # 1. Reclamo formal vs Sentimiento
    ct_reclamo = pd.crosstab(df["reclamo_formal"], df["sentimiento"], normalize="index")[orden] * 100
    ax = ct_reclamo.plot(kind="bar", stacked=True, color=colores, figsize=(8, 5), rot=0)
    for container in ax.containers:
        labels = [f"{v:.1f}%" if v >= 4 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")
    plt.title("Distribución de Sentimiento según Reclamo Formal")
    plt.xlabel("Reclamo Formal")
    plt.ylabel("Porcentaje de Reseñas (%)")
    plt.legend(title="Sentimiento", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("reports/figures/bivariado_reclamo_sentimiento.png", dpi=150)
    plt.close()

    # 2. Problema resuelto vs Sentimiento
    ct_problema = pd.crosstab(df["problema_resuelto"], df["sentimiento"], normalize="index")[orden] * 100
    ax = ct_problema.plot(kind="bar", stacked=True, color=colores, figsize=(8, 5), rot=0)
    for container in ax.containers:
        labels = [f"{v:.1f}%" if v >= 4 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")
    plt.title("Distribución de Sentimiento según Resolución del Problema")
    plt.xlabel("Problema Resuelto")
    plt.ylabel("Porcentaje de Reseñas (%)")
    plt.legend(title="Sentimiento", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("reports/figures/bivariado_problema_sentimiento.png", dpi=150)
    plt.close()

    # 3. Tiempo de respuesta (horas) vs Sentimiento
    paleta_dict = {
        "Negativo": "#d9534f",
        "Neutral": "#f0ad4e",
        "Positivo": "#5cb85c",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="sentimiento",
        y="tiempo_respuesta_horas",
        order=orden,
        hue="sentimiento",
        hue_order=orden,
        palette=paleta_dict,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    medianas = df.groupby("sentimiento")["tiempo_respuesta_horas"].median().reindex(orden)
    for idx, med in enumerate(medianas):
        ax.text(idx, med + 0.6, f"Mediana: {med:.1f} h", ha="center", fontweight="bold", color="black")
    plt.title("Tiempo de Respuesta (horas) por Clase de Sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Tiempo de Respuesta (horas)")
    plt.tight_layout()
    plt.savefig("reports/figures/bivariado_tiempo_respuesta_sentimiento.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    generar_visualizaciones()
    generar_graficos_bivariados()