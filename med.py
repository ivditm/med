from core.utils import (process_data,
                        make_full_analyses_anomalies,
                        anal_hyp, binary_search, Regressor)
import streamlit as st
from dotenv import load_dotenv
import os
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go


load_dotenv()
st.set_option('deprecation.showPyplotGlobalUse', False)


id_ = os.getenv('id')
gid = int(os.getenv('gid'))

id_ = st.secrets['id']
gid = st.secrets['gid']


st.title('Анализ медицинсткого центра')
st.markdown('''## Декомпозиция

### Задачи:

1. Базовый анализ данных
2. Подготовка данных
3. Исследовательский анализ данных
4. Определение основных показателей,
                 описывающих стоимость лечения, в рамках каждой гипотезы
5. Подсчет выручки за 2021 и 2022 годы, оценка изменений
6. Выдвижение гипотез
7. Описание показателей, которые требуется рассчитать для проверки гипотезы
8. Расчет выбранных показателей на доступных данных
9. Вывод о подтверждении или опровержении гипотезы на основе данных и расчет
                 влияния данного фактора на повышение средней стоимости
10. Общий вывод по результатам исследования

### Гипотезы:

1. Средняя стоимость обслуживания пациентов изменилась из-за
                 изменения цен на услуги
2. Средняя стоимость обслуживания изменилась из-за изменения интенсивности
                 лечения пациентов
3. Средняя стоимость обслуживания пациентов изменилась из-за изменения
                 половозрастной структуры обслуживаемых пациентов
''')


st.header('Предобработка')
data = process_data(id_, gid)
data['service_amount'] = data['service_amount'].apply(lambda x:
                                                      float(
                                                           (x
                                                            .replace('\xa0',
                                                                     '')
                                                            .replace(',', '.')
                                                            .strip())))
data['service_number'] = data['service_number'].astype(int)
data['age_for_service_date'] = data['age_for_service_date'].astype(int)
data['insured'] = data['insured'].astype(int)
data['sex_id'] = data['sex_id'].astype(int)
data['year'] = data['service_date'].dt.year.astype(str)
data['sex_id'] = data['sex_id'].map({1: 'муж', 2: 'жен'})
data['price'] = data['service_amount'] / data['service_number']
data['age_poligon'] = data['age_for_service_date'].apply(func=binary_search)
data['service_date'] = data['service_date'].dt.date
data['service_date'] = data['service_date'].map(str)
max_date = str(data.query('year=="2022"')['service_date'].max())
min_date = str(data.query('year=="2022"')['service_date'].min())
max_date_21 = '-'.join(['2021', *max_date.split('-')[1:]])
min_date_21 = '-'.join(['2021', *min_date.split('-')[1:]])
data_2021 = data.query('@min_date_21 <= service_date <= @max_date_21')
y = "2022"
data_2022 = data.query('year == @y')
data = pd.concat([data_2021, data_2022], ignore_index=True)

st.header('Основные показатели:')
year = st.radio('Выберите год', [2021, 2022])
year = str(year)
data_indicators = data.query('year == @year')
revenue = data_indicators['service_amount'].sum()
avg_check = round(revenue / len(data_indicators['insured'].unique()))
cnt_clients = len(data_indicators['insured'].unique())
cnt_services = len(data_indicators['service_name'].unique())
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'#### Выручка: {revenue}')
    st.markdown(f'#### Средний чек: {avg_check}')
with col2:
    st.markdown(f'#### Кол-во поситителей: {cnt_clients}')
    st.markdown(
        f'#### Кол-во уникальных оказываемых услуг: {cnt_services}')

effective = (data_indicators
             .groupby('insured', as_index=False)
             .agg({'service_name': 'count'})[['insured', 'service_name']]
             .rename(columns={'service_name': 'effectife'}))
effective = (effective
             .groupby(by='effectife',
                      as_index=False)
             .agg({'insured': 'count'})
             .rename(columns={'insured': 'num_clients'}))
q25, q75 = np.quantile(effective['effectife'], [0.25, 0.75])
effective['poligon_effectfes'] = (
    effective['effectife']
    .apply(
            lambda x:
            f'{effective["effectife"].min()}-{q25} визитов'
            if x < q25
            else (
             f'{q25}-{q75} визитов'
             if q25 <= x <= q75
             else
             f'{q75}-{effective["effectife"].max()} визитов'
                                  )
                                  ))
effective = (effective
             .groupby(by='poligon_effectfes', as_index=False)
             .agg({'num_clients': 'sum'}))
q25, q75 = np.quantile(data_indicators['price'], [0.25, 0.75])
data_indicators['poligon_price'] = (data_indicators['price']
                                    .apply(
    lambda x:
    f'{data_indicators["price"].min()}-{q25} руб'
    if x < q25
    else (
        f'{q25}-{q75} руб'
        if q25 <= x <= q75
        else
        f'{q75}-{data_indicators["price"].max()} руб'
    )
))
structure_price = (data_indicators
                   .groupby(by='poligon_price', as_index=False)
                   .agg({'service_name': 'count'})
                   .rename(columns={'service_name': 'effectives'}))

fig = go.Figure(
    data=[
        go.Pie(labels=effective['poligon_effectfes'],
               values=effective['num_clients'],
               hole=.3)
    ]
)
fig.update_layout(
    title_text="Интесивность лечения"
)
st.plotly_chart(fig)

fig = go.Figure(
    data=[
        go.Pie(labels=structure_price['poligon_price'],
               values=structure_price['effectives'],
               hole=.3)
    ]
)
fig.update_layout(
    title_text="Структура услуг по стоимости"
)
st.plotly_chart(fig)

with st.expander('Посмотреть данные'):
    st.dataframe(data)


st.header('Общий анализ')
column = st.selectbox('Выберите колонку для анализа', [
    "service_amount",
    "service_number",
    "age_for_service_date"
], key=0)
factor = st.selectbox('Выберете фактор', [
    None,
    "service_name",
    "sex_id",
    "year",
    "age_poligon"], key=1)
make_full_analyses_anomalies(data, column, factor)


fig = px.scatter_3d(
    data,
    x='service_amount',
    y='service_number',
    z='age_for_service_date',
    color='sex_id'
)
st.plotly_chart(fig)


factor = st.selectbox(
    'Выберете фактор',
    ['sex_id',
     'age_poligon'],
    key=2)
column = st.selectbox(
    'Выберете колонку',
    [
        "service_amount",
        "service_number",
        "age_for_service_date"])
sns.boxplot(data=data, x=column, y="year", hue=factor)
st.pyplot()


st.header('Анализ гипотез')
factor = st.selectbox(
    'Выбирете фактор',
    [
        "service_amount",
        "service_number",
        "age_for_service_date"
    ])
alpha = st.slider('Choose alpha level',
                  min_value=0.01,
                  max_value=1.0,
                  step=0.01,
                  value=0.05,
                  key=10)
st.subheader(f'Изменение {factor} по годам')
anal_hyp(
    data.query('year == "2021"')[factor],
    data.query('year == "2022"')[factor],
    alpha=alpha
)


st.subheader('Попробуем найти взаимосвязь между'
             ' ценами и возрастом/кол-вом визитов')
NUMERICAL_COLUMNS = ["service_number", "age_for_service_date"]
predictors = st.selectbox('Выберите предиктор:', NUMERICAL_COLUMNS)
graphs = st.selectbox('Выберите тип графика:', ['table', 'plot'])
reg = Regressor(
    data,
    predictors,
    "service_amount",
    [
        f'Взаимосвязь {predictors} и service_amount',
        predictors,
        "service_amount"
    ]
)
if graphs == 'table':
    st.write(reg.analysis())
elif graphs == 'plot':
    reg.get_plot()


st.header('Выводы:')
st.markdown("""В ходе проведенного исследования были
             выявлены статистически значимые изменения
             в следующих параметрах: возраст пациентов,
             количество пациентов и стоимость услуг.
             Эти выводы подкреплены результатами
            статистических тестов и визуальным
            анализом графиков.
Однако, несмотря на значимость
             обнаруженных изменений, регрессионный
            анализ показал, что эти изменения не
             были обусловлены внутренними факторами.
             Скорее всего, они являются результатом
             воздействия внешних факторов.
             Это указывает на необходимость
             дальнейшего изучения этих внешних
            факторов для более точного понимания
             их влияния на обслуживание пациентов
             и стоимость услуг.
""")
