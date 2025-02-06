################################################################################
# %% IMPORTAÇÕES
import geopandas as gpd # para trabalhar com os dados geográficos

from joblib import load # para importar o modelo

# import numpy as np
import pandas as pd

import streamlit as st # interface WEB - https://streamlit.io/

# arquivos utilizados
from notebooks.src.config import(
    DADOS_GEO_MEDIAN,
    DADOS_LIMPOS,
    MODELO_FINAL
)




################################################################################
# %% FUNÇÕES CACHE_DATA
# qualquer coisa que possa ser armazenado em database
# Python primitives, dataframe e API calls

@st.cache_data
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)

@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)




################################################################################
# %% FUNÇÕES CACHE_RESOURCE
# qualquer coisa que NÃO possa ser armazenado em database
# ML models e database connections

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)




################################################################################
# %% DADOS EM CACHE
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

################################################################################
# %% PAGINA

st.title('Previsão de preços de imóveis')

# inputs
longitude = st.number_input(label='Longitude', min_value=-124.30, max_value=-114.56, value=-122.33, format='%0.6f')
latitude = st.number_input(label='Latitude', min_value=32.54, max_value=41.96, value=37.88, format='%0.6f')

housing_median_age = st.number_input(label='Idade do imóvel', min_value=1, max_value=51, value=10, format='%d')

total_rooms = st.number_input(label='Total de cômodos', min_value=1, max_value=11026, value=800, format='%d')
total_bedrooms = st.number_input(label='Total de quartos', min_value=1, max_value=2205, value=100, format='%d')
population = st.number_input(label='População', min_value=1, max_value=5804, value=300, format='%d')
households = st.number_input(label='Domicílios', min_value=1, max_value=1979, value=100, format='%d')

median_income = st.slider(label='Renda média (múltiplos de US$ 10k)', min_value=0.0, max_value=11.0, value=4.5, step=0.5, format='%0.1f')

ocean_proximity = st.selectbox(label='Proximidade do oceano', options=sorted(df['ocean_proximity'].unique()), index=0,)

median_income_cat = st.number_input(label='Categoria de renda', min_value=1, max_value=5, value=4, format='%d')

rooms_per_household = st.number_input(label='Cômodos por domicílio', min_value=1.0, max_value=11.0, value=7.0, format='%0.2f')
bedrooms_per_room = st.number_input(label='Quartos por cômodo', min_value=0.0, max_value=5.0, value=0.2, format='%0.2f')
population_per_household = st.number_input(label='Pessoas por domicílio', min_value=0.0, max_value=6.0, value=2.0, format='%0.2f')

# True if the button was clicked on the last run of the app, False otherwise.
botao_previsao = st.button(label='Prever preço')

################################################################################
# %% construindo a entrada do modelo

if botao_previsao: # True se clicou no botão
    entrada_modelo = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity,
        'median_income_cat': median_income_cat,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
    }

    # df_entrada_modelo = pd.DataFrame(data=[entrada_modelo])
    df_entrada_modelo = pd.DataFrame(data=entrada_modelo, index=[0])

    preco = modelo.predict(df_entrada_modelo) # * 1E3 # faz a predição com os dados da tela

    st.write( # mostrando o preço
        f'Preço previsto: <b>US$ {preco[0][0]:,.2f}</b>'.replace('.', '¬').replace(',', '.').replace('¬', ','),
        unsafe_allow_html=True # torne True se quiser mostrar um HTML
    )

    # st.write( # mostrando o dataframe para validação
    #     df_entrada_modelo.T,
    #     unsafe_allow_html=False # torne True se quiser mostrar um HTML
    # )

################################################################################