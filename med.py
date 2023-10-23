from core.utils import (process_data,
                        make_full_analyses_anomalies,
                        anal_hyp, binary_search, Regressor)
import streamlit as st
from dotenv import load_dotenv
import os
import seaborn as sns
import plotly.express as px
import pandas as pd


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
data['sex_id'] = data['sex_id'].astype(int)
data['year'] = data['service_date'].dt.year.astype(str)
data['sex_id'] = data['sex_id'].map({1: 'муж', 2: 'жен'})
data['price'] = data['service_amount']/data['service_number']
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
avg_check = round(revenue/len(data_indicators['insured'].unique()))
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
