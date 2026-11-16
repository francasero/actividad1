import os
import warnings
import pandas as pd
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def generar_perfilado():
    output_dir = "reports/perfilado"
    os.makedirs(output_dir, exist_ok=True)

    df_clientes = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    df_resenas = pd.read_csv("data/raw/datos_resenas.csv", sep=";")

    perfil_clientes = ProfileReport(
        df_clientes,
        title="Perfilado de Datos Crudos - Clientes",
        minimal=True,
    )
    perfil_clientes.to_file(os.path.join(output_dir, "perfilado_clientes.html"))

    perfil_resenas = ProfileReport(
        df_resenas,
        title="Perfilado de Datos Crudos - Resenas",
        minimal=True,
    )
    perfil_resenas.to_file(os.path.join(output_dir, "perfilado_resenas.html"))

    print(f"Informes de ydata-profiling exportados en {output_dir}")


if __name__ == "__main__":
    generar_perfilado()