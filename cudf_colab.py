

!nvidia-smi ###La utilidad de línea de comandos nvidia-smi proporciona información adicional sobre el estado de la GPU.

"""### Alternativa se carga la extension %load_ext cudf.pandas para aumentar la velocidad de pandas, pertenece al proyecto Rapid"""

import pandas as pd
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
# %load_ext cudf.pandas
# pandas API is now GPU accelerated

import cudf
import cupy as cp
import os

cp.random.seed(0)  ###La función random. seed inicializa el generador de números aleatorios. a: Semilla a utilizar

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Corrected URL to access the raw content of the CSV file
# url = "https://raw.githubusercontent.com/VINY2030/datasets/refs/heads/main/SaratogaHouses.csv"
# 
# # Read the CSV file from the URL
# data = pd.read_csv(url) # Using the variable 'url' instead of the literal string 'url'

# Commented out IPython magic to ensure Python compatibility.
# %%time
# data

"""# **<font color="#07a8ed">Creando-`cudf.Series`**."""

s = cudf.Series([4, 5, 6, None, 8])
s

"""# **<font color="#07a8ed">cudf.DataFrame de 100 millones de registros?**."""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# np.random.seed(0)
# 
# num_rows = 100_000_000
# num_columns = 12
# 
# # Create a DataFrame with random data
# df_cudf = pd.DataFrame(np.random.randint(0, 100, size=(num_rows, num_columns)),
#                   columns=[f'Column_{i}' for i in range(1, num_columns + 1)])
# df_cudf

df_cudf.info()

df_cudf.isna().sum() ## registros na

for i in df_cudf.columns:
    print(i)
    print(df_cudf[i].nunique())

for i in df_cudf.columns:
    print(df_cudf[i].value_counts())

import plotly.express as px
import plotly.graph_objects as go

fig = px.parallel_coordinates(df_cudf, color="Column_1", labels={"Column_1": "Column_1", "Column_2": "Column_2", "Column_3": "Column_3"
    , "Column_4": "Column_4", "Column_5": "Column_5", "Column_6": "Column_6", "Column_7": "Column_7", "Column_8": "Column_8",
    "Column_9": "Column_9", "Column_10": "Column_10", "Column_11": "Column_11", "Column_12": "Column_12"},
                    color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
fig.show()

"""# **<font color="#07a8ed">Creando un cudf.DataFrame con valores especificos**."""

dfcudf = cudf.DataFrame(
    {
        "a": list(range(20)),
        "b": list(reversed(range(20))),
        "c": list(range(20)),
    }
)
dfcudf

"""##Ver las filas superiores de un marcador de datos GPU."""

pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
gdf = cudf.DataFrame.from_pandas(pdf)
gdf

dfcudf.head(2)

"""# **<font color="#07a8ed">Seleccionar una columna**."""

dfcudf["a"]

"""# **<font color="#07a8ed">Seleccionar por etiquetas**.

##Seleccionar filas del índice 2 al índice 5 de las columnas 'A' y 'B'.
"""

dfcudf.loc[2:5, ["a", "b"]]

"""# **<font color="#07a8ed">Seleccionar por posicion**.

##Seleccionar a través de enteros y slices, como Numpy/Pandas.
"""

dfcudf.iloc[0]

dfcudf.iloc[0:3, 0:2]

"""#También puede seleccionar elementos de un marco de datos o una serie con acceso de índice directo."""

dfcudf[3:5]

"""# **<font color="#07a8ed">Indexacion Booleana**.

##Seleccionar filas en un marco o serie de datos mediante indexación booleana directa.
"""

dfcudf[dfcudf.b > 15]

"""##Seleccionar valores de un `DataFrame` donde se cumple una condición booleana, a través de la API 'Query`."""

dfcudf.query("b == 3")

"""##Con CUDF estándar, puede usar la palabra clave local_dict o pasar directamente la variable a través de la palabra clave @. Los operadores lógicos compatibles incluyen>, <,> =, <=, == y! =."""

cudf_comparator = 3
dfcudf.query("b == @cudf_comparator")

"""##Usando el método ISIN para el filtrado."""

dfcudf[dfcudf.a.isin([0, 5])]

"""# **<font color="#07a8ed">Multiindice**.

##CUDF admite la indexación jerárquica de los marcos de datos utilizando Multiindex. La agrupación jerárquicamente (ver la agrupación a continuación) produce automáticamente un marco de datos con un multiíndex.
"""

arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
tuples = list(zip(*arrays))
idx = cudf.MultiIndex.from_tuples(tuples)
idx

"""###Este índice puede respaldar cualquier eje de un marcado de datos."""

gdf1 = cudf.DataFrame(
    {"first": cp.random.rand(4), "second": cp.random.rand(4)}
)
gdf1.index = idx
gdf1

gdf2 = cudf.DataFrame(
    {"first": cp.random.rand(4), "second": cp.random.rand(4)}
).T
gdf2.columns = idx
gdf2

"""##Acceso a los valores de un marco de datos con un multiíndex, ambos con .loc"""

gdf1.loc[("b", 3)]

"""e `.iloc`"""

gdf1.iloc[0:2]

"""# **<font color="#07a8ed">Missing Data**.

##Los datos faltantes se pueden reemplazar utilizando el método `Fillna`.
"""

s = cudf.Series([4, 5, 6, None, 8])
s

s.fillna(999)

"""# **<font color="#07a8ed">Estadistica**.

Calculando estadistica descriptiva de una serie
"""

s.describe()

s.mean(), s.var()

"""# **<font color="#07a8ed">ApplyMap**.

##Aplicar funciones a una `series`. Tenga en cuenta que la aplicación de funciones definidas por el usuario directamente con DASK-CUDF aún no se ha implementado. Por ahora, puede usar [MAP_PARTITIONS] (http://docs.doask.org/en/stable/generated/dask.dataframe.dataframe.map_partitions.html) para aplicar una función a cada partición de los datos distribuidos.
"""

def add_ten(num):
    return num + 10


dfcudf["a"].apply(add_ten)

"""# **<font color="#07a8ed">Histograma**.

https://docs.rapids.ai/visualization/

##Contando el número de ocurrencias de cada valor único de variable
"""

dfcudf.a.value_counts()

"""# **<font color="#07a8ed">Cadena de caracteres**.

##Al igual que los pandas, CUDF proporciona métodos de procesamiento de cadenas en el atributo STR de la serie. La documentación completa de los métodos de cadena es un trabajo en progreso. Consulte la documentación de la API CUDF para obtener más información.
"""

s = cudf.Series(["A", "B", "C", "Aaba", "Baca", None, "CABA", "dog", "cat"])
s.str.lower()

"""##Además de una simple manipulación, también podemos hacer coincidir las cadenas usando [expresiones regulares] (https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.core.column.string.stringmethods.match.html )."""

s.str.match("^[aAc].+")

"""# **<font color="#07a8ed">Concat**.

##Concatenando  `series` y` Dataframes` en cuanto a la fila.
"""

s = cudf.Series([1, 2, 3, None, 5])
s1= cudf.DataFrame({"s": [1, 2, 3, None, 5], "b": [10, 20, 30, 40, 50]})
s1 =cudf.concat([s, s])
s1

s2 =cudf.concat([s, s])
s2

s3= cudf.DataFrame(cudf.concat([s,s1]))

s3

type(s3)

"""# **<font color="#07a8ed">Join**.

##Realizar el estilo SQL que fusiona. Tenga en cuenta que el orden de DataFrame no se mantiene, pero puede restaurarse después de la fusión mediante la clasificación del índice.
"""

df_a = cudf.DataFrame()
df_a["key"] = ["a", "b", "c", "d", "e"]
df_a["vals_a"] = [float(i + 10) for i in range(5)]

df_b = cudf.DataFrame()
df_b["key"] = ["a", "c", "e"]
df_b["vals_b"] = [float(i + 100) for i in range(3)]

merged = df_a.merge(df_b, on=["key"], how="left")
merged

"""# **<font color="#07a8ed">Agrupacion**.

##Al igual que [Pandas] (https://pandas.pydata.org/docs/user_guide/groupby.html), Cudf y Dask-cudf admiten el paradigma de Groupby [split-apply-Combine Groupby] (https://doi.org/10.18637 /jss.v040.i01).
"""

## dfcudf

dfcudf["agg_col1"] = [1 if x % 2 == 0 else 0 for x in range(len(dfcudf))]
dfcudf["agg_col2"] = [1 if x % 3 == 0 else 0 for x in range(len(dfcudf))]

"""Agrupar y luego aplicar la función `Sum` a los datos agrupados."""

dfcudf.groupby("agg_col1").sum()

"""Agrupación jerárquicamente luego aplicando la función `Sum` a datos agrupados."""

dfcudf.groupby(["agg_col1", "agg_col2"]).sum()

"""Agrupar y aplicar funciones estadísticas a columnas específicas, utilizando `AGG`."""

dfcudf.groupby("agg_col1").agg({"a": "max", "b": "mean", "c": "sum"})

"""# **<font color="#07a8ed">Sorting**.

Clasificación por valores.
"""

dfcudf.sort_values(by="b")

"""# **<font color="#07a8ed">Transpose**.

Transposición de un marcador de datos, utilizando el método `Transpose` o la propiedad` T`. Actualmente, todas las columnas deben tener el mismo tipo. La transposición no se implementa actualmente en Dask-Cudf.
"""

sample = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
sample

sample.transpose()

"""# **<font color="#07a8ed">TimeSeries**.

`Dataframes` admite columnas escrita` DateTime`, que permiten a los usuarios interactuar y filtrar datos basados ​​en marcas de tiempo específicas.
"""

import datetime as dt
import cudf
import pandas as pd
import cupy as cp

date_df = cudf.DataFrame()
# Create the DatetimeIndex without 'freq'
dates = pd.date_range("11/20/2018", periods=72)
# Convert to cudf.Series and assign to column
date_df["date"] = cudf.Series(dates)
date_df["value"] = cp.random.sample(len(date_df))

search_date = dt.datetime.strptime("2018-11-23", "%Y-%m-%d")
date_df.query("date <= @search_date")

"""# **<font color="#07a8ed">Categoricals**.

`Dataframes` admite columnas categóricas.
"""

gdf = cudf.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
)
gdf["grade"] = gdf["grade"].astype("category")
gdf

"""Acceso a las categorías de una columna. Tenga en cuenta que esto no es compatible actualmente en Dask-Cudf."""

gdf.grade.cat.categories

"""Acceso a los valores del código subyacente de cada observación categórica."""

gdf.grade.cat.codes

"""# **<font color="#07a8ed">Conversiones de Datos**.

## Pandas

Convertir un CUDF `DataFrame` a un Pandas` DataFrame`.
"""

gdf.head().to_pandas()

"""## Numpy

Convertir un marco de datos CUDF o DASK-CUDF a una ndarray numpy.
"""

gdf.to_numpy()

"""Convertir un CUDF o dask-Cudf `series` a un` ndarray` numpy."""

gdf["grade"] = gdf["grade"].astype(str)
grade_array = gdf["grade"].to_numpy()

grade_array

gdf["id"].to_numpy()

"""## Arrow

https://pypi.org/project/pyarrow/

Convertir un marco de datos CUDF o DASK-CUDF a una tabla Pyarrow.
"""

gdf.to_arrow()

"""# **<font color="#07a8ed">Lectura de archivos**.

## CSV

Escribir en un archivo CSV.
"""

# Assuming you have data to create a DataFrame, replace this with your actual data
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)  # Create a DataFrame

if not os.path.exists("example_output"):
    os.mkdir("example_output")

df.to_csv("example_output/foo.csv", index=False)

df

"""Lectura de un archivo CSV."""

df = cudf.read_csv("example_output/foo.csv")
df

"""Tenga en cuenta que para el caso DASK-CUDF, utilizamos dask_cudf.read_csv con preferencia a dask_cudf.from_cudf (cudf.read_csv) ya que el primero puede paralelarse a través de múltiples GPU y manejar archivos CSV más grandes que caben en la memoria en una sola GPU.

Leer todos los archivos CSV en un directorio en un solo dask_cudf.dataframe, usando el comodín Star.

## Parquet

Escribir a los archivos de Parquet con el escritor de parquet acelerado de CUDF
"""

df.to_parquet("example_output/temp_parquet")
df

"""Lectura de archivos de parquet con el lector de parquet acelerado de GPU de CUDF."""

df = cudf.read_parquet("example_output/temp_parquet")
df

"""## ORC

Escribir archivos ORC.

####Los archivos ORC (Optimized Row Columnar) se usan para almacenar datos en sistemas de procesamiento de big data como Hadoop. Son un formato de almacenamiento en columnas que optimiza la lectura, escritura y procesamiento de datos
"""

df.to_orc("example_output/temp_orc")
df

"""y lectura"""

df2 = cudf.read_orc("example_output/temp_orc")
df2

"""# **<font color="#07a8ed">Rendimiento con gran marco de datos**.

CUDF es excelente para manejar grandes marcos de datos. En este ejemplo agregamos valores después de agrupar por una clave:
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext cudf.pandas
# pandas API is now GPU accelerated

import cudf
import cupy as cp
import os

cp.random.seed(0)  ###La función random. seed inicializa el generador de números aleatorios. a: Semilla a utilizar

# Commented out IPython magic to ensure Python compatibility.
nr = 100_000_000
df = cudf.DataFrame({
    'key': cp.random.randint(0, 10, nr),
    'value': cp.random.random(nr)
})

# %time df.groupby('key')['value'].mean()

pdf = df.to_pandas()
# %time pdf.groupby('key')['value'].mean()

import cudf
import cupy as cp

# Assuming df is your cuDF DataFrame
total_estimated_size = df.memory_usage('bytes').sum()
print(f"Total estimated size of the DataFrame: {total_estimated_size} bytes")

"""CUDF también tiene algoritmos de unión eficientes. En este ejemplo, usamos una unión hash para combinar valores de dos marcos de datos basados ​​en una clave:"""

# Commented out IPython magic to ensure Python compatibility.
nr = 50_000_000
df = cudf.DataFrame({
        'key': cp.random.randint(0, 10, nr),
        'value': cp.random.random(nr)
})
lookup = cudf.DataFrame({
        'key': range(10),
        'lookup': cp.random.random(10)
})

# %time df.merge(lookup, on='key')

pdf = df.to_pandas()
plookup = lookup.to_pandas()
# %time pdf.merge(plookup, on='key')

"""La computación y la aplicación de filtros en los marcos de datos CUDF también son operaciones eficientes."""

# Commented out IPython magic to ensure Python compatibility.
nr = 20_000_000
df = cudf.DataFrame({
    'rating_a': cp.random.randint(1, 5, nr),
    'rating_b': cp.random.randint(1, 5, nr),
    'rating_c': cp.random.randint(1, 5, nr),
})

# %time df.where(df>3)

pdf = df.to_pandas()
# %time pdf.where(pdf>3)

"""Y la clasificación es otra tarea donde CUDF muestra una gran aceleración."""

# Commented out IPython magic to ensure Python compatibility.
nr = 50_000_000
df = cudf.DataFrame({
    'a': cp.random.rand(nr),
    'b': cp.random.rand(nr),
    'c': cp.random.rand(nr),
})

# %time df.sort_values('a')

pdf = df.to_pandas()
# %time pdf.sort_values('a')