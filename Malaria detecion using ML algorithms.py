import pandas as pd
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import app as st
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.write("""
Malaria disease prediction using ML algorithms
"""
)
# Location of image named 'Malaria-mosquito'
image=Image.open('C:/Malaria.jpg')
st.image(image,caption='ML web application',use_column_width=True)
# Location of dataset
mal1=pd.read_csv('C:/dataset.csv')

# mal1=df.drop('age',axis=1)
values = {"fever":{"no":0, "yes":1},"cold":{"no":0, "yes":1},"rigor":{"no":0, "yes":1},"fatigue":{"no":0, "yes":1},
          "headace":{"no":0, "yes":1},"bittertongue":{"no":0, "yes":1},"vomitting":{"no":0, "yes":1},
          "diarrhea":{"no":0, "yes":1},"Convolusion":{"no":0, "yes":1},"Anemia":{"no":0, "yes":1},"jaundice":{"no":0, "yes":1},
          "cocacola_urine":{"no":0, "yes":1},"hypoglycemia":{"no":0, "yes":1},"prostraction":{"no":0, "yes":1},
          "hyperpyrexia":{"no":0, "yes":1},"sex":{"Male":0,"Female":1},"Malaria":{"yes": 1,"no": 0}}
mal1.replace(values,inplace=True)
st.subheader('Following is a table representing values of dataset we are using in 0 or 1.')
st.dataframe(mal1)
st.write("Value 0 is for No and 1 for Yes ")
st.write("For Male the value is 0 and for female it is 1")
st.write("Given below is a table consisting mathematical operations on our input values.")
st.write(mal1.describe())
y=mal1['Malaria']
mal=mal1.drop('Malaria',axis=1)
x=mal.iloc[:,0:16].values
a=''

def get_user_input():

    fever=st.sidebar.slider('fever',0,1,0)
    cold=st.sidebar.slider('cold',0,1,1)
    rigor=st.sidebar.slider('rigor',0,1,0)
    fatigue=st.sidebar.slider('faigue',0,1,1)
    headace=st.sidebar.slider('headace',0,1,0)
    bittertongue=st.sidebar.slider('bitter tongue',0,1,1)
    vomitting=st.sidebar.slider('vomitting',0,1,0)
    diarrhea=st.sidebar.slider('diarrhea',0,1,1)
    Convolusion=st.sidebar.slider('Convolusion',0,1,1)
    Anemia=st.sidebar.slider('Anemia',0,1,0)
    jaundice=st.sidebar.slider('jaundice',0,1,0)
    cocacola_urine=st.sidebar.slider('cocacola_urine',0,1,0)
    hypoglycemia=st.sidebar.slider('hypoglycemia',0,1,1)
    prostraction=st.sidebar.slider('prostraction',0,1,0)
    hyperpyrexia=st.sidebar.slider('hyperpyrexia',0,1,0)
    sex=st.sidebar.slider('sex',0,1,0)
    user_data={'fever': fever,'cold': cold,'rigor':rigor,'fatigue':fatigue,'headace':headace,'bittertongue':bittertongue,
               'vomitting':vomitting,'diarrhea':diarrhea,'Convolusion':Convolusion,
               'Anemia':Anemia,'jaundice':jaundice,'cocacola_urine':cocacola_urine,'hypoglycemia':hypoglycemia,
               'prostraction':prostraction,'hyperpyrexia':hyperpyrexia,'sex':sex

               }
    features = pd.DataFrame(user_data, index=[0])
    return features

set = st.selectbox("Choose the model you by which want to get your result: ", ('LogisticsRegression','KNN','RandomForest'))
user_input = get_user_input()
st.subheader('User input')
st.write(user_input)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

u=0
if(set=='LogisticsRegression'):
   u=0
elif(set=='KNN'):
    u=1
else:
    u=2


if(u==0):
    # LogisticsRegression algo
    logreg = LogisticRegression(C=1).fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    user = logreg.predict(user_input)
    st.write("Accuracy of the LogisticsRegression algorithm is: ", logreg.score(x_test, y_test)*100,'%')
elif(u==1):
    #     KNN algorithm
    my_knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    my_knn.fit(x, y)
    y_pred = my_knn.predict(x_test)
    user=my_knn.predict(user_input)
    st.write('Accuracy of KNN algo is: ', metrics.accuracy_score(y_test, y_pred) * 100,'%')
else:
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(x_train,y_train)
    y_pred = rfc.predict(x_test)
    user=rfc.predict(user_input)
    st.write("Accuracy of RandomForest algo is: ", metrics.accuracy_score(y_test, y_pred)*100,'%')

if(user==1):
    st.write('OOPS!!! You are infected with malaria.Please visit an expert as soon as possible.')
else:
    st.write('Congrats...You are safe from malaria.')

