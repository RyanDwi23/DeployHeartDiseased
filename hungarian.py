#Import Library
import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle



# Fungsi untuk mengatur tampilan Streamlit
#st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Fungsi untuk memanggil dataset 'Hungarian.data'
with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

#Kode itertools.takewhile digunakan untuk mengambil elemen-elemen dari suatu iterable (dalam konteks ini, tampaknya daftar lines) sampai suatu kondisi tertentu terpenuhi.
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

#pembentukan DataFrame, beberapa operasi pengolahan data dilakukan. Pertama, hanya kolom-kolom tertentu yang dipilih untuk analisis selanjutnya dengan menggunakan indeks tertentu. Kemudian, beberapa kolom tertentu dihilangkan dari DataFrame untuk tujuan analisis lebih lanjut. Setelah itu, semua nilai dalam DataFrame diubah menjadi tipe data float untuk konsistensi. Terakhir, semua nilai -9.0 dalam DataFrame digantikan dengan NaN untuk menandai nilai yang hilang atau tidak valid.
df = pd.DataFrame.from_records(data)
df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)
df.replace(-9.0, np.NaN, inplace=True)
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

#Perintah untuk labeling
column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

#df_selected.rename(columns=column_mapping, inplace=True) mengubah nama kolom dari sebuah dataframe yang disimpan dalam variabel df_selected berdasarkan pemetaan kolom yang didefinisikan sebelumnya dalam column_mapping. 
#Selanjutnya, columns_to_drop = ['ca', 'slope','thal'] mendefinisikan daftar nama kolom yang perlu dihapus dari dataframe. Kemudian, menggunakan df_selected = df_selected.drop(columns_to_drop, axis=1), kolom-kolom yang disebutkan dalam columns_to_drop dihapus dari dataframe df_selected. 
df_selected.rename(columns=column_mapping, inplace=True)
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

#variabel-variabel seperti meanTBPS, meanChol, meanfbs, meanRestCG, meanthalach, dan meanexang 
#digunakan untuk menghitung rata-rata dari kolom-kolom tertentu dalam df_selected setelah menghapus nilai yang hilang (NaN) dari masing-masing kolom tersebut.
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

#Mengubah tipe data menjadi float
meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

#menghitung rata-rata dari beberapa variabel kardiovaskular seperti tekanan darah, kolesterol, gula darah, denyut jantung maksimal, angina yang diinduksi oleh latihan, 
#dan elektrokardiogram istirahat, lalu membulatkannya ke bilangan bulat terdekat.
meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

#menciptakan sebuah kamus (dictionary) yang berisi rata-rata nilai dari beberapa variabel kardiovaskular yang sebelumnya dihitung, 
#yang akan digunakan untuk mengisi nilai-nilai yang hilang dalam data set yang sedang diproses.
fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

#mengisi nilai-nilai yang hilang dalam dataframe yang telah dipilih dengan rata-rata yang telah dihitung, menghapus duplikat baris, melakukan oversampling menggunakan teknik SMOTE untuk menyeimbangkan kelas target, 
#dan akhirnya memuat model prediksi dari file yang disimpan sebelumnya.
df_clean = df_selected.fillna(value=fill_values)
df_clean.drop_duplicates(inplace=True)
X = df_clean.drop("target", axis=1)
y = df_clean['target']
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
model = pickle.load(open("combination_model.pkl", 'rb'))

#melakukan prediksi menggunakan model yang telah dimuat, menghitung akurasi prediksi terhadap data aktual, 
#dan menyimpan hasil prediksi beserta label target ke dalam dataframe akhir.
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title="Hungarian Heart Disease",
  page_icon=":heart:"
)

# Menambahkan style untuk mengatur warna background
st.markdown(
    """
    <style>
        body {
            background-color: #F5DD9B !important;
        }
    </style>
    """, unsafe_allow_html=True
)



# Header dengan warna dan efek teks
st.markdown("<h1 style='text-align: center; color: #333366;'>Hungarian Heart Disease</h1>", unsafe_allow_html=True)

st.image("https://img.freepik.com/free-vector/doctor-with-stethoscope-listening-huge-heart-beat-ischemic-heart-disease_335657-4397.jpg?w=996&t=st=1704173675~exp=1704174275~hmac=0f712a91cf89a887e75df001cbe07e2b56b7159dd56890783fc07a5d43841b76", width=500)
# Menampilkan akurasi model dengan warna dan format yang menarik
st.markdown(f"**_Model's Accuracy_**: <span style='color: green; font-weight: bold;'>{accuracy}%</span> <span style='color: red;'>[_Do not copy outright_]</span>", unsafe_allow_html=True)
st.text("------------------------------------------------------------------------------------------------")
st.markdown(f"**Nur Ryan Dwi Cahyo**", unsafe_allow_html=True)
st.markdown(f"**A11.2020.12610**", unsafe_allow_html=True)
st.text("------------------------------------------------------------------------------------------------")

#Membuat dua tab pada tampilan dashboard
tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

#Konfigurasi pada tab pertama
with tab1:
  #Judul
  st.sidebar.header("**User Input** Sidebar")
  
  # untuk memungkinkan pengguna memasukkan nilai usia (age) dengan batasan minimum dan maksimum yang berasal dari dataframe df_final.
  age = st.sidebar.number_input(label=":violet[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")

  #memungkinkan pengguna memilih jenis kelamin (sex) melalui dropdown sidebar di Streamlit, lalu mengkonversi pilihan tersebut menjadi nilai numerik dengan 1 untuk "Male" dan 0 untuk "Female".
  sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  #memungkinkan pengguna memilih tipe nyeri dada (chest pain type) melalui dropdown sidebar di Streamlit dan mengkonversi pilihan tersebut menjadi nilai numerik, dengan kategorisasi yang telah ditentukan dari 1 hingga 4 berdasarkan jenis nyeri dada yang dipilih.
  cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic


  # input angka pada sidebar Streamlit untuk memungkinkan pengguna memasukkan tekanan darah istirahat (resting blood pressure) dengan batasan minimum dan maksimum yang diberikan oleh nilai minimal dan maksimal dari kolom trestbps dalam dataframe df_final.
  trestbps = st.sidebar.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['trestbps'].min()}**], :red[Max] value: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")

  chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['chol'].min()}**], :red[Max] value: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")

  fbs_sb = st.sidebar.selectbox(label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  #memungkinkan pengguna memilih hasil elektrokardiograf (electrocardiographic results) istirahat melalui dropdown sidebar di Streamlit dan 
  #mengonversi pilihan tersebut menjadi nilai numerik berdasarkan kategori yang telah ditentukan.
  restecg_sb = st.sidebar.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  #memungkinkan pengguna memasukkan nilai denyut jantung maksimal (maximum heart rate achieved) melalui input angka pada sidebar Streamlit dengan batasan minimum dan maksimum yang diberikan oleh kolom thalach dalam dataframe df_final, serta memungkinkan pengguna memilih apakah terdapat angina yang diinduksi oleh latihan fisik melalui dropdown sidebar, yang kemudian dikonversi menjadi nilai biner 0 atau 1.
  thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['thalach'].min()}**], :red[Max] value: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")

  exang_sb = st.sidebar.selectbox(label=":violet[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  #nput angka pada sidebar Streamlit untuk memungkinkan pengguna memasukkan nilai depresi ST yang diinduksi oleh latihan fisik relatif terhadap istirahat, dengan batasan minimum dan maksimum yang diberikan oleh kolom oldpeak dalam dataframe df_final.
  oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{df_final['oldpeak'].min()}**], :red[Max] value: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")

  #mengumpulkan semua data masukan yang telah dimasukkan oleh pengguna ke dalam sebuah kamus (dictionary), lalu mengonversinya menjadi sebuah DataFrame dengan satu baris dan menampilkannya dalam dua bagian terpisah untuk presentasi yang lebih rapi melalui antarmuka Streamlit.
  data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
  }

  preview_df = pd.DataFrame(data, index=['input'])

  st.header("User Input as DataFrame")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  result = ":violet[-]"

  predict_btn = st.button("**Predict**", type="primary")

  st.write("")

  #mengambil input dari pengguna untuk sejumlah variabel kesehatan, memprediksi status penyakit jantung berdasarkan model yang telah dilatih, menampilkan status prediksi dengan indikator kemajuan dan hasilnya dalam antarmuka Streamlit.
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]"

  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

#Konfigurasi pada tab kedua
with tab2:
  #judul
  st.header("Predict multiple data:")

  #mengambil lima baris pertama dari dataframe df_final, mengonversinya ke format CSV, dan menyediakan opsi bagi pengguna untuk mengunduh file CSV contoh atau mengunggah file CSV melalui antarmuka Streamlit.
  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  #membaca file CSV tersebut ke dalam dataframe, memprediksi status penyakit jantung untuk setiap baris menggunakan model yang telah dilatih, dan menampilkan status proses prediksi dalam antarmuka Streamlit.
  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    #mengkonversi setiap prediksi menjadi kategori status penyakit jantung, memvisualisasikan hasil prediksi dalam bentuk dataframe dan menampilkan dataframe yang diunggah oleh pengguna di antarmuka Streamlit dalam dua kolom yang berbeda.
    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
    

#Memberikan text copyright
st.markdown("<p style='text-align: center; color: #777;'>Copyright © 2024 by Nur Ryan Dwi Cahyo. All rights reserved.</p>", unsafe_allow_html=True)

