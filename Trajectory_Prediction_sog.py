#!/usr/bin/env python
# coding: utf-8

# # <h3 align="center">Prédiction de trajectoire</h3>
# ___
# Ce notebook contient un code modifier venant d'hugging face dans lequel on cherche à prédire la trajectoire des navires dans une zone donnée, ici la Méditerranée grâce aux données AIS.

# ![image.png](attachment:88691f04-0d7f-43cf-8f5a-6d856cf0854b.png)

# #### <font color='red'></font>

# #### <font color='red'><h3 align="center">Installation des bibliothèques nécessaire pour faire fonctionner ce notebook</h3></font>
# ___

# In[ ]:


pip install folium


# In[ ]:


pip install geopandas


# In[ ]:


pip install tokenizers


# In[ ]:


pip install datasets


# In[ ]:


pip install transformers


# In[ ]:


pip install accelerate -U


# In[ ]:


pip install psycopg


# In[ ]:


pip install seaborn


# In[ ]:


pip install scikit-learn


# In[ ]:


pip install haversine


# In[ ]:


pip install geopandas


# #### <font color='red'><h3 align="center">Lecture du fichier contenant les données AIS. </h3></font>
# ___

# In[ ]:


import psycopg
import seaborn as sns
import folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import geopandas as gp
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline
import numpy as np
import geopandas
torch.cuda.is_available()


# In[ ]:


df = pd.read_parquet("C:\\Users\\evan.da_costa_pina\\Documents\\Prediction_stage_evan\\test_vscode_cuda\\messages_ais_avril_2019_autoroute.parquet").rename(columns={"@timestamp": "timestamp"})
df.head()


# <br/>
# <br/>
# <br/>
# ___
# Définition du seuil de filtrage afin de faciliter le changement de ces valeurs.
# ___

# In[ ]:


sog_max_value = 40
sog_min_value = -1

cog_min_value = -1

mmsi_number_min = 5


# ## <h3 align="center">Pré-Analyse des données</h3>
# On analyse les données du DataFrame afin de retirer les données pouvant perturber l'apprentissage du modèle et refaire un DataFrame sans erreur.
# ___
# <br/>
# <br/>

# In[ ]:


df_max_time = df['timestamp'].max()
df_min_time = df['timestamp'].min()
print("timestamp max = ", df_max_time)
print("timestamp min = ", df_min_time)

df_max_sog = df['sog'].max()
df_min_sog = df['sog'].min()
print("Vitesse maximal = ", df_max_sog)
print("Vitesse minimal = ", df_min_sog)

count_max_sog = (df['sog'] >= sog_max_value).sum()
count_min_sog = (df['sog'] == sog_min_value).sum()
print("Nombre de navires avec une vitesse supérieure à ", sog_max_value," : ",count_max_sog)
print("Nombre de navires avec une vitesse égale à -1 : ", count_min_sog)


# ___
# Avec ces simples commandes on remarque dans un premier temps que les données ce trouvent sur une plage horraire de 24H. On observe également des vitesses trop élevées de la normal qui pourrait interférer dans l'apprentissage mais aussi un sog minimal à -1 qui peut fortement embrouiller le modèle (si on lui donne le sog pour son apprentissage bien sur). 
# ___
# <br/>
# <br/>

# In[ ]:


df_max_cog = df['cog'].max()
df_min_cog = df['cog'].min()
print("Cog maximal = ", df_max_cog)
print("Cog minimal = ", df_min_cog)

count_min_cog = (df['cog'] == df_min_cog).sum()
print(f"Nombre de navires avec un angle égale à {df_min_cog} : ", count_min_cog)


# ___
# On observe un absurdité avec des angles négatifs qu'il faudra donc retirer de notre DataFrame.
# ___
# <br/>
# <br/>

# In[ ]:


df.groupby("mmsi").size().describe()


# In[ ]:


mmsi_counts = df['mmsi'].value_counts()

count_mmsi_min = (mmsi_counts < mmsi_number_min).sum()

print("Nombre de bateaux qui envoie moins de 5 fois sa position : ", count_mmsi_min)


# ___
# Ici on observe que 564 navires émettent moins de 5 signale ce qui peut être trop faible pour l'entraînement du modèle.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


df_max = df.max(axis=0, skipna=True, numeric_only=False)
df_min = df.min(axis=0, skipna=True, numeric_only=False)

print("Tableau des valeurs max : ")
print(df_max)
print("Tableau des valeurs min : ")
print(df_min)


# In[ ]:


mmsi_to_remove = mmsi_counts[mmsi_counts < mmsi_number_min].index

df_update_mmsi = df[~df['mmsi'].isin(mmsi_to_remove)]

print(f"Nombre de lignes avant filtration : {len(df)}")
print(f"Nombre de lignes après filtration : {len(df_update_mmsi)}")


# ___
# On supprime tous les mmsi des navires qui emméttaient moins de 5 signale.
# ___
# <br/>
# <br/>

# In[ ]:


df_update_mmsi2 = df_update_mmsi.replace({"sog": sog_min_value, "cog": cog_min_value},np.NaN).reset_index(drop=True)
df_update_mmsi2


# In[ ]:


df_update = df_update_mmsi2[df_update_mmsi2['sog'] <= sog_max_value].reset_index(drop=True)
df_update


# In[ ]:


df_update.shape


# ___
# On vérifie le nombre de colonne et de ligne.
# ___
# <br/>
# <br/>

# In[ ]:


df_update.groupby("mmsi").filter(lambda x : len(x) > 30 and len(x) < 100).groupby("mmsi").size()


# #### <font color='red'><h3 align="center">Visualisation</h3></font>
# ___

# 
# Dans cette section, nous allons donner un aspect visuel à notre DataFrame pour essayer d'obtenir des informations.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


df["cog"].plot.hist(bins=20)
df_update["cog"].plot.hist(bins=20)


# ___
# On constate en bleu un très gros pique en 0 qui est du aux valeurs absurdes avec les -1 qui se "cachent" en 0.
# Et on observe également en orange notre dataframe corrigée.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


df.dtypes


# In[ ]:


df["sog"].plot.hist(bins=20)
#df_update["sog"].plot.hist(bins=20)


# ___
# On remarque également que certains bateaux peuvent avoir des vitesses trop élevées et qu'on pourrait enlever dans les données à fournir au modèle.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


df_update["sog"].plot.hist(bins=20)


# ___
# On remarque bien que la dataframe update n'a plus de vitesses trop élevées.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


sns.pairplot(df_update)


# ___
# On regarde ici si des relations sont possibles entre les différentes données du DataFrame.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


sns.relplot(x="lon", y="lat", size="sog",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df_update)


# In[ ]:


df_lon_lat = df_update.select_dtypes(include=['int','float'])[["lat","lon"]]
df_lon_lat.head()


# In[ ]:


df_sog = df_update.select_dtypes(include=['int','float'])[["sog"]]
df_sog.head()


# ___
# On filtre notre DataFrame pour ne prendre que les colonnes qui nous intérressent et faciliter l'utilisation du Kmeans.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


groupes_kmeans_lon_lat = KMeans(n_clusters=4, random_state=0).fit_predict(df_lon_lat)
groupes_kmeans_lon_lat


# In[ ]:


groupes_kmeans_sog = KMeans(n_clusters=4, random_state=0).fit_predict(df_sog)
groupes_kmeans_sog


# In[ ]:


groupes_kmeans_cog = KMeans(n_clusters=4, random_state=0).fit_predict(df_update.select_dtypes(include=['int','float'])[["cog"]].dropna())
groupes_kmeans_cog


# ___
# On paramètre nos Kmeans afin de faire un cluster sur les positions des navires et un autre sur les vitesses des navires
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


localisation = folium.Map(location=[40.221233, 6.051645], zoom_start = 5.55)


# In[ ]:


positions_lon_lat = pd.DataFrame({'latitude' : df_update['lat'], 'longitude' : df_update['lon'], 'cluster' : groupes_kmeans_lon_lat})


# In[ ]:


liste_couleurs = ["red","green","blue", "pink"]
colors = [liste_couleurs[c] for c in positions_lon_lat['cluster']]
for i,c in enumerate(liste_couleurs):
    plt.plot([], [], marker='o', color=c, label=i, linestyle="")
plt.scatter(positions_lon_lat['longitude'], positions_lon_lat['latitude'], marker='o', color=colors)
plt.legend(loc=(1.1,0))
plt.title("Classification de position", color = "skyblue")


# In[ ]:


localisation = folium.Map(location=[40.221233, 6.051645], zoom_start = 5.55)

groupes = positions_lon_lat.groupby("cluster")

for i, (name, group) in enumerate(groupes):
    for long,lat in zip(group["longitude"], group["latitude"]):
        folium.Circle(location=[lat, long],
                      radius=1000,
                      color=liste_couleurs[i],
                      fill=True,
                      fill_color=liste_couleurs[i],
                      fill_opacity =1).add_to(localisation)

localisation


# In[ ]:


positions_sog = pd.DataFrame({'latitude' : df_update['lat'], 'longitude' : df_update['lon'], 'cluster' : groupes_kmeans_sog})


# In[ ]:


liste_couleurs = ["red","green","blue", "pink"]
colors = [liste_couleurs[c] for c in positions_sog['cluster']]
for i,c in enumerate(liste_couleurs):
    plt.plot([], [], marker='o', color=c, label=i, linestyle="")
plt.scatter(positions_sog['longitude'], positions_sog['latitude'], marker='o', color=colors)
plt.legend(loc=(1.1,0))
plt.title("Classification par sog", color = "skyblue")


# In[ ]:


positions_cog = pd.DataFrame({'latitude' : df_update.dropna(subset="cog" )['lat'], 'longitude' : df_update.dropna(subset="cog")['lon'], 'cluster' : groupes_kmeans_cog})
liste_couleurs = ["red","green","blue", "pink"]
colors = [liste_couleurs[c] for c in positions_cog['cluster']]
for i,c in enumerate(liste_couleurs):
    plt.plot([], [], marker='o', color=c, label=i, linestyle="")
plt.scatter(positions_cog['longitude'], positions_cog['latitude'], marker='o', color=colors)
plt.legend(loc=(1.1,0))
plt.title("Classification par cog", color = "skyblue")


# In[ ]:


loclisation = folium.Map(location=[40.221233, 6.051645], zoom_start = 5.55)

groupes = positions_sog.groupby("cluster")

for i, (name, group) in enumerate(groupes):
    for long,lat in zip(group["longitude"], group["latitude"]):
        folium.Circle(location=[lat, long],
                      radius=1000,
                      color=liste_couleurs[i],
                      fill=True,
                      fill_color=liste_couleurs[i],
                      fill_opacity =1).add_to(localisation)

loclisation


# ___
# Affichage des clusters sur la carte
# ___
# <br/>
# <br/>
# <br/>

# #### <font color='red'><h3 align="center">Entraînement du modèle</h3></font>
# ___

# In[ ]:


from tqdm import tqdm
def get_trajectory_str(df_update, mmsi=None):
    result = []
    ext_df = df_update[['timestamp', 'lat', 'lon', 'mmsi']]
    
    if mmsi is not None:
        mmsi_list = [mmsi]
    else:
        mmsi_list = ext_df['mmsi'].unique()
    
    for mmsi in mmsi_list:
        a = ext_df[ext_df['mmsi'] == mmsi].sort_values(by='timestamp')[['timestamp', 'lat', 'lon']]
        result.append("".join([" # " + str(a.iloc[i].values[0]) + ", " + str(a.iloc[i].values[1]) + ", " + str(a.iloc[i].values[2]) for i in range(a.shape[0])]))
    
    return result

data_str = get_trajectory_str(df_update)
data_str[0]


# ___
# L'objectif de cette fonction est de convertir certaines données du DataFrame en texte brut, ici un ou tous les mmsi.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


df = pd.DataFrame(data_str, columns=["traj"])
dataset = Dataset.from_pandas(df)
dataset


# In[ ]:


df["traj"].apply(lambda x: len(x)).describe()


# ___
# Ici on transfome le texte brut donné par la fonction "get_str_traj" en une nouvelle DataFrame qui aura un colonne "Traj" et ou chaque ligne correspondra a la trajectoire d'un navire unique.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
tokenizer


# ___
# Chargement du tokenizer : Le tokenizer pré-entraîné est chargé pour transformer les textes en tokens.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


context_length = 1024

def tokenize(element):
    outputs = tokenizer(
        element["traj"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=["traj"]) # attention il y aurait un nombre diff de features entre traj et input ids sans le remove columns
tokenized_datasets


# ___
# ### Cette fonction tokenize :
# <br/>
# Prend un élément contenant une chaîne de texte sous la clé "traj".
# <br/>
# Utilise un tokenizer pour convertir cette chaîne en une ou plusieurs séquences de tokens, tronquant et gérant les débordements si nécessaire.
# <br/>
# Filtre les séquences de tokens pour ne conserver que celles qui ont exactement context_length tokens.
# <br/>
# Retourne un dictionnaire contenant ces séquences de tokens filtrées sous la clé "input_ids".

# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")


# ___
# Ce code configure et initialise un modèle GPT-2 avec des paramètres spécifiés, puis calcule et affiche la taille du modèle en paramètres.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets[i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")


# ___
# Ce code initialise un DataCollator pour le modèle de langage, remplace le token de padding par le token de fin de séquence, puis applique ce collator à un échantillon de 5 éléments tokenisés pour afficher leurs formes respectives.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


tokenized_datasets= tokenized_datasets.train_test_split(test_size=0.1) #get an eval dataset
tokenized_datasets


# ___
# Division du dataset en ensemble d'entraînement et de test.
# ___
# <br/>
# <br/>
# <br/>

# ___
# ### <u>Définition des arguments pour l'entrainement<u>
# Ce code spécifie les hyperparamètres et les configurations pour l'entraînement. Cela inclut des paramètres comme la taille du lot, le taux d'apprentissage, la stratégie d'évaluation, etc.
# ### <u>Création de l'objet Trainer<u>
# on y initialise un Trainer avec le modèle, le tokenizer, les arguments de formation, le collateur de données, et les jeux de données d'entraînement et d'évaluation.
# ### <u>Utilité de ce code<u>
# Ce code ci-dessous configure et initialise un environnement d’entraînement pour un modèle de langage. Le Trainer de transformers libère l’utilisateur de nombreuses tâches d’entraînement, telles que les boucles d’entraînement, les évaluations périodiques, la journalisation, et la gestion des taux d’apprentissage, en utilisant des valeurs par défaut raisonnables. Bien que j’utilise ici des valeurs par défaut, le Trainer est incroyablement personnalisable. Vous pouvez manipuler les arguments de formation pour contraindre ou laisser libres n’importe lequel des hyperparamètres afin de voir comment le modèle se comporte.

# In[ ]:


args = TrainingArguments(
    output_dir="/content/sample_data/traj",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=25,
    logging_steps=25,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=1000,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)


# In[ ]:


args.device


# In[ ]:


trainer.train()


# ___
# Fonction qui démarre l'entrainement du modèle
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


#torch.save(model.state_dict(), 'C:/Users/Da Costa Pina/Documents/esiee/Stage/Rendu/model.pth')
#torch.save(model.state_dict(), "C:\\Users\\victor.litoux\\Documents\\test_vscode_cuda\\mode_max-l1024_26_06_2024_v2.pth")


# #### <font color='red'><h3 align="center">Fonction pour utiliser le modèle entraîné</h3></font>
# ___

# Dans cette section, nous détaillons les fonctions qui permettent de charger le modèle, de l'utiliser ainsi que de prédire une trajectoire pour un navire spécifique.
# <br/>
# <br/>
# ___

# In[ ]:


def load_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer


# ___
# cette fonction encapsule le chargement du modèle pré-entraîné GPT-2 et du tokenizer à partir des chemins spécifiés.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


def use_model(model, tokenizer, input_text, max_length, max_new_tokens=50, num_return_sequences=1):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device, 
    )

    generated_text = pipe(input_text, max_new_tokens=256, num_return_sequences=num_return_sequences, pad_token_id=pipe.tokenizer.eos_token_id)
    return generated_text


# ___
# Fonction pour utiliser le modèle entrainé.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


segment_input_start = int(context_length/60)
segment_input_start


# In[ ]:


def predict_traj(model, tokenizer, mmsi, dfsog_cog, context_length=1024, max_new_tokens=50):

    df_mmsi = dfsog_cog[dfsog_cog['mmsi']==mmsi].sort_values("timestamp")
    df_input = df_mmsi[-segment_input_start:-5]
    df_last = df_mmsi[-5:]
    traj_str: str = get_trajectory_str(df_input, mmsi)[0][-context_length:]
    
    prediction = use_model(model, tokenizer, traj_str, max_length=1024, num_return_sequences=1)
    
    return prediction[0]["generated_text"].removeprefix(traj_str),traj_str

model_path = 'C:\\Users\\evan.da_costa_pina\\Documents\\Prediction_stage_evan\\test_vscode_cuda\\mode_max-l1024_26_06_2024_v2.pth'
tokenizer_path = 'huggingface-course/code-search-net-tokenizer'
model, tokenizer = load_model(model_path, tokenizer_path)

mmsi = 235113769
predicted_traj,traj_input = predict_traj(model, tokenizer, mmsi, df_update)

traj_input,predicted_traj


# ___
# Cette fonction retourne la trajectoire prédite et la trajectoire d'entrée utilisée pour la prédiction.
# ___
# <br/>
# <br/>
# <br/>

# #### <font color='red'><h3 align="center">Affichage et validation du modèle (ou pas)</h3></font>
# ___

# In[ ]:


predicted_traj[:int(1024*0.9)],predicted_traj[int(1024*0.9):]


# In[ ]:


str_output = str(predicted_traj[2:int(1024*0.9)])
indesirable = ['"', '[', ']', '{', '}', "'",'generated_text', ': ']
for char in indesirable :
    str_output = str_output.replace(char, '')
    
print(str_output)


# In[ ]:


def text_to_dataframe(str_output):
    data = []
    for text in str_output:
        lines = text.strip().split(" # ")
        for line in lines:
            if line.strip():
                row = line.split(',')

                if len(row) == 2:  # Pour la première ligne avec lat et lon seulement
                    
                    lat = row[0].strip()
                    lon = row[1].strip()
                    timestamp = pd.NaT
                    data.append([timestamp, lat, lon])
                elif len(row) == 3:  # Pour les autres lignes avec timestamp, lat et lon
                    timestamp = row[0].strip()
                    lat = row[1].strip()
                    lon = row[2].strip()
                    data.append([timestamp, lat, lon])
    df = pd.DataFrame(data, columns=['timestamp', 'lat', 'lon'])

    df.replace('nan', np.nan, inplace=True)  # Remplacer les chaînes 'nan' par np.nan

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    return df

df_predicted = text_to_dataframe([str_output])
df_predicted


# In[ ]:


mmsi = 235113769
df_mmsi = df_update[df_update['mmsi']==mmsi].sort_values("timestamp")
df_input = df_mmsi[-segment_input_start:-5]
dfunique = df_input[df_input['mmsi'] == mmsi]
df_real = dfunique[-5:].sort_values("timestamp").drop(['type_new', 'sog','cog', 'mmsi'], axis=1)
df_real


# In[ ]:


real_list_of_tuples= [tuple(r) for r in df_real[['lat', 'lon']].to_numpy()]
real_list_of_tuples


# In[ ]:


real_list_lon = df_real['lon'].to_list()
real_list_lat = df_real['lat'].to_list()

print(real_list_lon)
print(real_list_lat)


# In[ ]:


df_predicted = text_to_dataframe([str_output])
df_predicted


# In[ ]:


predicted_list_of_tuples= [tuple(r) for r in df_predicted[['lat', 'lon']].to_numpy()]
predicted_list_of_tuples


# In[ ]:


predicted_list_lon = df_predicted['lon'].to_list()
predicted_list_lat = df_predicted['lat'].to_list()

print(predicted_list_lon)
print(predicted_list_lat)


# In[ ]:


df_update.groupby("mmsi").filter(lambda x : len(x) > 100 and len(x) < 150).groupby("mmsi").size()


# In[ ]:


from folium import Popup
def plot_circle_in_color(df_in: pd.DataFrame, color:str,medit, mmsi : int):
    for lon, lat in zip(df_in['lon'], df_in['lat']):
        folium.Circle(
            location=[lat, lon],
            popup = Popup(f"mmsi : {mmsi}"),
            radius=2000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5            
        ).add_to(medit)
        coordinates = list(zip(df_in['lat'], df_in['lon']))
        folium.PolyLine(locations=coordinates, color=color).add_to(medit)

def predict_and_plot_traj(list_mmsi: [int], df_in: pd.DataFrame):
    medit = folium.Map(location=[32.8291 ,-178], zoom_start=5)
    for mmsi in tqdm(list_mmsi):
        predicted_traj,traj_input =predict_traj(model, tokenizer, mmsi, df_in)
        str_output = str(predicted_traj[2:int(1024*0.9)])
        indesirable = ['"', '[', ']', '{', '}', "'",'generated_text', ': ']
        for char in indesirable :
            str_output = str_output.replace(char, '')
        df_predicted = text_to_dataframe([str_output])
        dfunique = df_in[df_in['mmsi'] == mmsi]
        df_input = dfunique[-segment_input_start:-5]
        df_real = dfunique[-5:]
        plot_circle_in_color(df_input, "blue",medit, mmsi)
        plot_circle_in_color(df_predicted.dropna(), "darkred",medit, mmsi)
        plot_circle_in_color(df_real,"green",medit, mmsi)
    return medit,df_predicted

list_of_mmsi = [232012089,
235113769,
246302000,
255806172,
258964000,
305717000,
311000149,
311000744,
352383000,
370594000,
538004693,
538005779,
538006131,
210998000,
232011018,
235114185,
303031000,
303656000,
305908000,
311000317,
356958000,
477486300,
538005476,
636015248]

map, df_predicted = predict_and_plot_traj(list_of_mmsi ,df_update)
map


# ___
# La trajectoire complète du navire est indiquée par la suite de points bleus, tandis que la trajectoire prédite est indiquée par la suite de points rouges.
# ___
# <br/>
# <br/>
# <br/>

# In[ ]:


if len(df_real) < len(df_predicted) :
    longueur_list = len(df_real)
elif len(df_real) > len(df_predicted) :
    longueur_list = len(df_predicted)


# In[ ]:


from haversine import haversine, Unit

def performance_pred(real_list_of_tuples, predicted_list_of_tuples):
    distances = []
    
    for i in range(longueur_list):
        real_lat, real_lon = real_list_of_tuples[i]
        pred_lat, pred_lon = predicted_list_of_tuples[i]

        distance = haversine((real_lat, real_lon), (pred_lat, pred_lon), unit=Unit.KILOMETERS)
        distances.append(distance)
    
    return distances


mmsi = 235113769
df_mmsi = df_update[df_update['mmsi']==mmsi].sort_values("timestamp")
df_input = df_mmsi[-segment_input_start:-5]
dfunique = df_input[df_input['mmsi'] == mmsi]
df_real = dfunique[-5:].sort_values("timestamp").drop(['type_new', 'sog','cog', 'mmsi'], axis=1)
real_list_of_tuples= [tuple(r) for r in df_real[['lat', 'lon']].to_numpy()]


#df_predicted = text_to_dataframe([str_output])
predicted_list_of_tuples= [tuple(r) for r in df_predicted[['lat', 'lon']].to_numpy()]


result = performance_pred(real_list_of_tuples, predicted_list_of_tuples)
print(result)


# In[ ]:


def create_df_distance (real_list_of_tuples, predicted_list_lat, predicted_list_lon, result):
    df_compare = pd.DataFrame(real_list_of_tuples, columns = ['Latitude_réel','Longitude_réel'])
    df_compare.loc[:, "Latitude_prédite"] = predicted_list_lat[:longueur_list]
    df_compare.loc[:, "Longitude_prédite"] = predicted_list_lon[:longueur_list]
    df_compare.loc[:, "Distance_km"] = result
    return df_compare
create_df_distance (real_list_of_tuples, predicted_list_lat, predicted_list_lon, result)


# In[ ]:


"""from sklearn.metrics import mean_squared_error
import numpy as np

test= []

for i in range(longueur_list) :
    long_lat_pred = [predicted_list_lon[i], predicted_list_lat[i]]
    long_lat = [real_list_lon[i], real_list_lat[i]]
    
     
    MSE_long_lat = np.square(np.subtract(long_lat, long_lat_pred)).mean(axis=0)

    
    RMSE_long_lat = 1- np.sqrt(MSE_long_lat)
    test.append(RMSE_long_lat)


test"""


# In[ ]:


"""1 - np.sqrt(np.square(df_compare["Latitude_réel"]-df_compare["Latitude_prédite"]).mean(axis=0))"""


# In[ ]:


#max_distance = 50 #float(input("Donner une distance maximum : "))
#min_distance  = 1 #float(input("Donner une distance minimum : "))

def score_prediction(dataframe_distance, max_distance, min_distance, seuil):
    distance_list = dataframe_distance.to_list()
    list_resultat_distance = []
    somme = 0
    
    for i in range(longueur_list) :        
        if distance_list[i] > max_distance:
            list_resultat_distance.append(0)
            somme += list_resultat_distance[i]
            
        elif distance_list[i] < min_distance:
            list_resultat_distance.append(1)
            somme += list_resultat_distance[i]
            
        else :
            list_resultat_distance.append(np.exp((-(distance_list[i])**2)/seuil**2))
            somme += list_resultat_distance[i]
            
    print(list_resultat_distance)
    moyenne = somme/longueur_list
    #print(moyenne)
    return list_resultat_distance

df_compare = create_df_distance(real_list_of_tuples, predicted_list_lat, predicted_list_lon, result)
score_prediction(df_compare['Distance_km'], 50, 1, 10)


# In[ ]:


from folium import Popup
def plot_circle_in_color(df_update: pd.DataFrame, color:str,medit, mmsi : int):
    for lon, lat in zip(df_update['lon'], df_update['lat']):
        folium.Circle(
            location=[lat, lon],
            popup = Popup(f"mmsi : {mmsi}"),
            radius=2000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5            
        ).add_to(medit)
        coordinates = list(zip(df_update['lat'], df_update['lon']))
        folium.PolyLine(locations=coordinates, color=color).add_to(medit)

def prediction_of_all (list_mmsi: [int], df_update: pd.DataFrame) :
    medit = folium.Map(location=[32.8291 ,-178], zoom_start=5)
    list_dataframe = []
    for mmsi in list_mmsi:
        print(f"Score de précision pour le mmsi {mmsi} : ")
        predicted_traj, traj_input =predict_traj(model, tokenizer, mmsi, df_update)
        str_output = str(predicted_traj[2:int(1024*0.9)])
        indesirable = ['"', '[', ']', '{', '}', "'",'generated_text', ': ']
        for char in indesirable :
            str_output = str_output.replace(char, '')
        df_predicted = text_to_dataframe([str_output])
        dfunique = df_update[df_update['mmsi'] == mmsi]
        df_input = dfunique[-segment_input_start:-5]
        df_real = dfunique[-5:]
        plot_circle_in_color(df_input, "blue",medit, mmsi)
        plot_circle_in_color(df_predicted.dropna(), "darkred",medit, mmsi)
        plot_circle_in_color(df_real,"green",medit, mmsi)
        real_list_lon = df_real['lon'].to_list()
        real_list_lat = df_real['lat'].to_list()
        real_list_of_tuples= [tuple(r) for r in df_real[['lat', 'lon']].to_numpy()]
        predicted_list_lon = df_predicted['lon'].to_list()
        predicted_list_lat = df_predicted['lat'].to_list()
        predicted_list_of_tuples= [tuple(r) for r in df_predicted[['lat', 'lon']].to_numpy()]
        if len(df_real) < len(df_predicted) :
            longueur_list = len(df_real)
        elif len(df_real) > len(df_predicted) :
            longueur_list = len(df_predicted)
        result = performance_pred(real_list_of_tuples, predicted_list_of_tuples)
        df_compare = create_df_distance (real_list_of_tuples, predicted_list_lat, predicted_list_lon, result)
        score_final = score_prediction(df_compare['Distance_km'], 50, 1, 10)
        df_compare.loc[:, "Score"] = score_final[:longueur_list]
        df_compare["mmsi"] = mmsi
        list_dataframe.append(df_compare)
    return medit, pd.concat(list_dataframe)

list_of_mmsi = [232012089,
235113769,
246302000,
255806172,
258964000,
305717000,
311000149,
311000744,
352383000,
370594000,
538004693,
538005779,
538006131,
210998000,
232011018,
235114185,
303031000,
303656000,
305908000,
311000317,
356958000,
477486300,
538005476,
636015248]

map, list_dataframe = prediction_of_all(list_of_mmsi ,df_update)
map


# In[ ]:


df_predicted[df_predicted["mmsi"] == 538006131]


# In[ ]:


list_dataframe[list_dataframe["mmsi"] == 235113769]


# In[ ]:




