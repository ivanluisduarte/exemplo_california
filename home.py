################################################################################
# %% IMPORTAÇÕES
import geopandas as gpd # para trabalhar com os dados geográficos

# from joblib import load # para importar o modelo

# import numpy as np
from numpy  import digitize, inf as infinito

import pydeck as pdk
import shapely
import streamlit as st # interface WEB - https://streamlit.io/

# arquivos utilizados
from notebooks.src.config import(
    DADOS_GEO_MEDIAN,
#     MODELO_FINAL
)


################################################################################
# %% FUNÇÕES CACHE_DATA
# qualquer coisa que possa ser armazenado em database
# Python primitives, dataframe e API calls

@st.cache_data
def carregar_dados_geo():
    # import geopandas as gpd
    # from notebooks.src.config import DADOS_GEO_MEDIAN
    # import shapely

    # return gpd.read_parquet(DADOS_GEO_MEDIAN) # as camadas não funcionan sem o tratamento abaixo criado pelo Chat GPT
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Fix invalid geometry
        # Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )
    
    # Apply the coordinate conversion and store in a new column
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo






@st.cache_data
def get_nomes_condados():
    return sorted(gdf_geo['name'].unique())


################################################################################
# %% FUNÇÕES CACHE_RESOURCE
# qualquer coisa que NÃO possa ser armazenado em database
# ML models e database connections

@st.cache_resource
def carregar_modelo():
    from joblib import load
    from notebooks.src.config import MODELO_FINAL

    return load(MODELO_FINAL)

################################################################################
# %% carregando arquivos ou cache

gdf_geo = carregar_dados_geo()
condados = get_nomes_condados()
modelo = carregar_modelo()

################################################################################
# %% PAGINA

st.title('Previsão de preços de imóveis')

# dividindo a tela em 2 colunas
coluna1, coluna2 = st.columns(spec=(0.35, 0.65), gap='small')


with coluna1:
    # inputs

    with st.form(
        key='formulario',
        clear_on_submit=False,
        border=False,
    ):
        selecionar_condados = st.selectbox(label='Condado', options=condados)
        housing_median_age = st.number_input(label='Idade do imóvel', min_value=1, max_value=50, value=10, format='%d')
        median_income = st.slider(label='Renda média anual (milhares de US$)', min_value=5, max_value=100, value=45, step=5, format='%d')

        st.form_submit_button(label='Prever preço e atualizar gráfico')


################################################################################
# %% construindo a entrada do modelo

    df_valores_condado = gdf_geo[gdf_geo['name'] == selecionar_condados].reset_index() # melhor para dataframes menores, que é o caso...

    #if botao_previsao: # True se clicou no botão  
    df_valores_condado.loc[0, 'housing_median_age'] = housing_median_age

    median_income /= 10 # os valores estão multiplicados por 10 mil, e estamos convertendo para 1 mil na apresentação
    df_valores_condado.loc[0, 'median_income'] = median_income

    median_income_cat = digitize(x=median_income, bins=[0, 1.5, 3, 4.5, 6, infinito], right=False)
    df_valores_condado.loc[0, 'median_income_cat'] = median_income_cat

    # faz a predição com os dados da tela
    preco = modelo.predict(df_valores_condado)

    st.metric( # mostrando o preço
        label='Preço previsto (US$)',
        value= f'{preco[0][0]:,.2f}'.replace('.', '¬').replace(',', '.').replace('¬', ','),
    )


################################################################################
# %% constuindo o mapa


with coluna2:

    tooltip = {
        'html': '<b>Condado:</b> {name}',
        'style': {
            'backgroundcolor': 'steelblue',
            'color': 'white',
            'fontsize': '10px',
        },
    }

    # colore o estado da califórnia
    polygon_layer = pdk.Layer(
        type='PolygonLayer',
        data=gdf_geo[['name', 'geometry']],
        get_polygon='geometry',
        get_fill_color=[0, 0, 255, 100], # RGB + alfa
        get_line_color=[255, 255, 255],
        get_line_width=50,
        pickable=True, # necessário para funcionar o tooltip
        auto_highlight = True,
    )

    # colore de cor diferente o condado selecionado
    highlight_layer = pdk.Layer(
        type='PolygonLayer',
        data=df_valores_condado[['name', 'geometry']],
        get_polygon='geometry',
        get_fill_color=[255, 0, 0, 100], # RGB + alfa
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True, # necessário para funcionar o tooltip
        auto_highlight = True,
    )

    # localização inicial
    initial_view_state = pdk.ViewState(
        latitude = float(df_valores_condado.loc[0, 'latitude']),
        longitude = float(df_valores_condado.loc[0, 'longitude']),
        zoom=4,
        min_zoom=3,
        max_zoom=10,
    )

    # mapa
    mapa = pdk.Deck(
        initial_view_state=initial_view_state,
        map_style='light',
        layers=[
            polygon_layer,
            highlight_layer,
        ],
        tooltip=tooltip,
    )

    # plotando o mapa
    st.pydeck_chart(
        pydeck_obj=mapa
    )


################################################################################