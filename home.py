################################################################################
# %% IMPORTAÇÕES
import geopandas as gpd # para trabalhar com os dados geográficos

from joblib import load # para importar o modelo

import numpy as np
import pandas as pd

import pydeck as pdk
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

@st.cache_data
def get_nomes_condados():
    return gdf_geo['name'].sort_values()


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
condados = get_nomes_condados()
modelo = carregar_modelo()

################################################################################
# %% PAGINA

st.title('Previsão de preços de imóveis')

# dividindo a tela em 2 colunas
coluna1, coluna2 = st.columns(spec=(0.35, 0.65), gap='large')


with coluna1:
    # inputs

    selecionar_condados = st.selectbox(label='Condado', options=condados)
    # longitude = st.number_input(label='Longitude', min_value=-124.30, max_value=-114.56, value=-122.33, format='%0.6f')
    # latitude = st.number_input(label='Latitude', min_value=32.54, max_value=41.96, value=37.88, format='%0.6f')

    housing_median_age = st.number_input(label='Idade do imóvel', min_value=1, max_value=50, value=10, format='%d')

    # total_rooms = st.number_input(label='Total de cômodos', min_value=1, max_value=11026, value=800, format='%d')
    # total_bedrooms = st.number_input(label='Total de quartos', min_value=1, max_value=2205, value=100, format='%d')
    # population = st.number_input(label='População', min_value=1, max_value=5804, value=300, format='%d')
    # households = st.number_input(label='Domicílios', min_value=1, max_value=1979, value=100, format='%d')



    median_income = st.slider(label='Renda média anual (milhares de US$)', min_value=5, max_value=100, value=45, step=5, format='%d')
    # median_income_cat = st.number_input(label='Categoria de renda', min_value=1, max_value=5, value=4, format='%d')

    # ocean_proximity = st.selectbox(label='Proximidade do oceano', options=sorted(df['ocean_proximity'].unique()), index=0,)


    # rooms_per_household = st.number_input(label='Cômodos por domicílio', min_value=1.0, max_value=11.0, value=7.0, format='%0.2f')
    # bedrooms_per_room = st.number_input(label='Quartos por cômodo', min_value=0.0, max_value=5.0, value=0.2, format='%0.2f')
    # population_per_household = st.number_input(label='Pessoas por domicílio', min_value=0.0, max_value=6.0, value=2.0, format='%0.2f')

    # True if the button was clicked on the last run of the app, False otherwise.
    botao_previsao = st.button(label='Prever preço')

    ################################################################################
    # %% construindo a entrada do modelo

    if botao_previsao: # True se clicou no botão
        # colunas que virão do GeoDataFrame
        colunas_condado = [
            'total_rooms', 'total_bedrooms', 'population', 'households',
            'ocean_proximity', 'rooms_per_household', 'bedrooms_per_room',
            'population_per_household', 'latitude', 'longitude', # 'centroid', 
        ]

        # array_valores_condado = gdf_geo.query(expr="name == @selecionar_condados")[colunas_condado].values[0] # mais legível e melhor para grandes dataframes
        array_valores_condado = gdf_geo[gdf_geo['name'] == selecionar_condados][colunas_condado].values[0] # melhor para dataframes menores, que é o caso...

        entrada_modelo = {k: v for k, v in zip(colunas_condado, array_valores_condado)}
        
        median_income /= 10 # os valores estão multiplicados por 10 mil, e estamos convertendo para 1 mil na apresentação
        # bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(x=median_income, bins=[0, 1.5, 3, 4.5, 6, np.inf], right=False)

        entrada_modelo.update({
            # 'latitude': entrada_modelo['centroid'].x,
            # 'longitude': entrada_modelo['centroid'].y,
            'housing_median_age': housing_median_age,
            'median_income': median_income,
            'median_income_cat': median_income_cat,
            # 'rooms_per_household': rooms_per_household,
            # 'bedrooms_per_room': bedrooms_per_room,
            # 'population_per_household': population_per_household,
        })

        # df_entrada_modelo = pd.DataFrame(data=[entrada_modelo])
        df_entrada_modelo = pd.DataFrame(data=entrada_modelo, index=[0])

        preco = modelo.predict(df_entrada_modelo) # * 1E3 # faz a predição com os dados da tela

        st.write( # mostrando o preço
            f'Preço previsto: <b>US$ {preco[0][0]:,.2f}</b>'.replace('.', '¬').replace(',', '.').replace('¬', ','),
            # df_entrada_modelo.T, # mostrando o dataframe para validação
            # entrada_modelo,
            unsafe_allow_html=True, # torne True se quiser mostrar um HTML
        )


################################################################################



with coluna2:

    # centroid = gdf_geo[gdf_geo['name'] == selecionar_condados]['centroid'].values[0]
    latitude, longitude  = gdf_geo[gdf_geo['name'] == selecionar_condados][['latitude', 'longitude']].values[0]

    # localização inicial
    initial_view_state = pdk.ViewState(
        # latitude=centroid.y,
        # longitude=centroid.x,
        latitude=float(latitude),
        longitude=float(longitude),
        zoom=5,
        min_zoom=4,
        max_zoom=10,
        pitch=50,
    )

    # mapa
    mapa = pdk.Deck(
        initial_view_state=initial_view_state,
        map_style='light',
    )

    # plotando o mapa
    st.pydeck_chart(
        pydeck_obj=mapa
    )


################################################################################