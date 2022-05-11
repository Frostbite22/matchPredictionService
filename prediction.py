from flask import Flask, json, g, request, jsonify
from flask_cors import CORS
import joblib
import os.path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import random
from IPython.display import Image
import pickle 
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)

### Data analysis Part 
df_match = pd.read_csv("results.csv")

years = []
for date in df_match.date:
   years.append(int(str(date)[0:4]))

#Making a new dataset with required features to train the machine learning model
#Year,Played Country,Team_1,team_2,team_1 score,team_2 score

New_Dataset_part_1=pd.DataFrame(list(zip(years,df_match.values[:,7],df_match.values[:,1],df_match.values[:,2],df_match.values[:,3],df_match.values[:,4])),columns=["year","Country","team_1","team_2","team_1_score","team_2_score"])
#Making a new dataset by changing the team_1 and team_2 and their respective scores
New_Dataset_part_2=pd.DataFrame(list(zip(years,df_match.values[:,7],df_match.values[:,2],df_match.values[:,1],df_match.values[:,4],df_match.values[:,3])),columns=["year","Country","team_1","team_2","team_1_score","team_2_score"])
New_Dataset=pd.concat([New_Dataset_part_1,New_Dataset_part_2],axis=0)
New_Dataset =New_Dataset.sample(frac=1).reset_index(drop=True) #Shaffling the dataset

#Creating a list containg all the names of the countries

teams_1=New_Dataset.team_1.unique()
contries=New_Dataset.Country.unique()
all_countries=np.unique(np.concatenate((teams_1,contries), axis=0))

#Defining the features and labels(Targets)

Y= New_Dataset.iloc[:,4:6] #Training targets (team_1_score and team_2_score)
categorized_data=New_Dataset.iloc[:,0:4].copy() #Traing features

label_encoder = preprocessing.LabelEncoder()

#Labeling the data using LabelEncorder in Sklearn-(Giving a unique number to each string(country))

label_encoder.fit(all_countries)
#list(label_encoder.classes_)
categorized_data['team_1']=label_encoder.transform(categorized_data['team_1'])
categorized_data['team_2']=label_encoder.transform(categorized_data['team_2'])
categorized_data['Country']=label_encoder.transform(categorized_data['Country'])

#Converting these feature columns to categrize form to make the training processs more smoother
categorized_data['team_1']=categorized_data['team_1'].astype("category")
categorized_data['team_2']=categorized_data['team_2'].astype("category")
categorized_data['Country']=categorized_data['team_2'].astype("category")


def select_winning_team(probability_array):
   prob_lst=[round(probability_array[0][i],3) for i in range(2)]
   if (prob_lst[0]>prob_lst[1]):
      out=0
   elif (prob_lst[0]<prob_lst[1]):
      out=1
   elif (prob_lst[0]==prob_lst[1]):
      out=2
   return out,prob_lst

   
###Making the model 
X=categorized_data

#Use any algorithm
model = MultiOutputRegressor(RandomForestClassifier())
model.fit(X,Y)


@app.route("/predict", methods=["POST"])
def predict():
   teams_json = json.loads(request.data)

   #Sample Prediction : here I need to pass the teams from post request

   year=2022
   stadium="Qatar"


   team_lst=[teams_json["team_1"],teams_json["team_2"]]
   team_1_num=label_encoder.transform([teams_json["team_1"]])[0]
   team_2_num=label_encoder.transform([teams_json["team_2"]])[0]
   stadium_num=label_encoder.transform([stadium])[0]

   #Sample Prediction Output

   X_feature=np.array([[year,stadium_num,team_1_num,team_2_num]])

   res=model.predict(X_feature)
   win,prob_lst=select_winning_team(res)

   return jsonify({f'{teams_json["team_1"]}' : f'{prob_lst[0]}',
                  f'{teams_json["team_2"]}' : f'{prob_lst[1]}'})

