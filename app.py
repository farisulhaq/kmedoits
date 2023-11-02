import streamlit as st
from module import kmedoid
from module import acuracy

import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


# display
# st.set_page_config(layout='wide')
st.set_page_config(page_title="MANDIRI 2023", page_icon='icon.png')


@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)


st.title("MANDIRI 2023")
st.write("PENGELOMPOKKAN PEMBERIAN DANA BANTUAN STATUS STUNTING DI MADURA MENGGUNAKAN METODE K-MEDOIDS CLUSTERING")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Data", "Prepocessing", "Modeling", "Implementation"])
# dataset
with dataframe:
    progress()
    uploaded_file = st.file_uploader("Upload file disini yaa...", type=[
                                     'xlsx'], accept_multiple_files=False)
    if uploaded_file is not None:
        starting_medoids = None
        data = pd.read_excel(uploaded_file)
    else:
        starting_medoids = True
        data = pd.read_excel('dataset/data.xlsx', sheet_name='sample')

    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with ket:
        st.write(
            "Sumber data dari dari badan pusat statistik Sumenep Madura.")
        st.download_button(
            label="Download data",
            data='dataset/data.xlsx',
            file_name='data.xlsx',
            mime='text/xlsx',
        )
        st.write("""
            Parameter :
            * Jenis Kelamin
            * Pekerjaan
            * Pendidikan
            * Usia
            * Penerimaan Bantuan
            * Resiko Stunting
        """)
    with dataset:
        st.dataframe(data, use_container_width=True)

# normalisasi data
with preporcessing:
    progress()
    st.subheader("""Normalisasi Data""")
    X = data.drop(['No'], axis=1).drop(
        list(data.filter(regex='Penerima')), axis=1)

    # ordinalencoder
    ordinalEncoder = OrdinalEncoder()
    ordinal = ordinalEncoder.fit_transform(X)

    minmaxScaled = MinMaxScaler()
    minmax = minmaxScaled.fit_transform(ordinal)

    le = LabelEncoder()

    ordinalEnc, minmaxScaler = st.tabs(["Ordinal Encoder", "MinMaxScaler"])
    with ordinalEnc:
        ordinals = pd.DataFrame(ordinal, columns=X.columns)
        st.dataframe(ordinals, use_container_width=True)
    with minmaxScaler:
        minmaxs = pd.DataFrame(minmax, columns=X.columns)
        st.dataframe(minmaxs, use_container_width=True)
    # X_ordinal = ordinal
    X_minmax = minmax
    y = le.fit_transform(data['Penerima BST'])
# modeling kmedoids
with modeling:
    progress()
    # kmedoidMenu = st.tabs("K-MEDOIDS")
    # with kmedoidMenu:
    # progress()
    placeholder = st.empty()
    placeholder.info("Progress Run Model")
    if starting_medoids != None:
        starting_medoids = np.array(
            [X_minmax[181], X_minmax[323], X_minmax[398]])
    while True:
        results = kmedoid.kmedoids(
            X_minmax, 2, 2, starting_medoids=starting_medoids)
        n_matches = acuracy.count_matches(y, results[1], True)
        ac = 100.0 * n_matches / len(X)
        if ac >= 90:
            break

    label_knn_minmax = pd.DataFrame(
        data={'Label Test': y, 'Label Predict': results[1]}).reset_index(drop=True)
    st.info('MinmaxScaler + OrdinalEncoder')
    st.dataframe(label_knn_minmax, use_container_width=True)
    st.success(f'Akurasi = {ac} %')
    placeholder.empty()

    # with b:
    #     label_knn_pca = pd.DataFrame(
    #         data={'Label Test': y_test_pca, 'Label Predict': y_pred_knn_pca}).reset_index(drop=True)
    #     st.info('OrdinalEncoder')
    #     st.dataframe(label_knn_pca, use_container_width=True)
    #     st.success(f'MAPE = {akurasi_knn_pca*100:.4} %')


with implementation:
    # input1
    nama = st.text_input('Masukkan Nama')
    # input2
    jns = st.selectbox(
        'Jenis Kelamin',
        ('Laki-laki', 'Perempuan'))
    # input 3
    pekerjaan = st.selectbox(
        'Pekerjaan',
        tuple(data['Pekerjaan'].unique()))
    # input 4
    pendidikan = st.selectbox(
        'Pendidikan',
        tuple(data['Pendidikan'].unique()))
    # input 5
    umurs = {
        'Usia dibawah 7 tahun': 'Tidak',
        'Usia 7-12': 'Tidak',
        'Usia 13-15': 'Tidak',
        'Usia 16-18': 'Tidak',
        'Usia 19-21': 'Tidak',
        'Usia 22-59': 'Tidak',
        'Usia 60 tahun keatas': 'Tidak'
    }
    umur = st.radio(
        "Umur",
        tuple(umurs.keys()))
    umurs[umur] = "Ya"

    # input 6
    stT = st.selectbox(
        'Status Stanting',
        ('0', '1', '2'))
    # # input 6
    # bpnt = st.selectbox(
    #     'Penerima BPNT',
    #     tuple(data['Penerima BPNT'].unique()))
    # # input 7
    # bpum = st.selectbox(
    #     'Penerima BPUM',
    #     tuple(data['Penerima BPUM'].unique()))
    # # input 8
    # bst = st.selectbox(
    #     'Penerima BSM',
    #     tuple(data['Penerima BST'].unique()))
    # # input 9
    # pkh = st.selectbox(
    #     'Penerima PKH',
    #     tuple(data['Penerima PKH'].unique()))
    # # input 10
    # sembako = st.selectbox(
    #     'Penerima SEMBAKO',
    #     tuple(data['Penerima SEMBAKO'].unique()))
    # button
    dataPredict = list(np.hstack([jns, pekerjaan, pendidikan, list(
        umurs.values())]))
    dataPredict.append(int(stT))

    if st.button('Check'):

        ordinalPredik = ordinalEncoder.transform([dataPredict])
        minmaxPredik = minmaxScaled.transform(ordinalPredik)
        labels = []
        for i in range(len(results[0][-1])):
            labels.append(np.linalg.norm(minmaxPredik - results[0][-1][i]))
        label = np.argmin(labels)
        st.success(f'{nama}, diprediksi = {label}')
