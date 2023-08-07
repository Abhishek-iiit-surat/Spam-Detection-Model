# IMPORTING THE IMPORTANT DEPENDENCIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
from sklearn import metrics
#READING THE DATA IN CSV FILE
raw_mail_data=pd.read_csv('./mail_data.csv', encoding='latin-1')
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)), '')
 

mail_data.loc[mail_data['Category']=='spam', 'Category',]=0
mail_data.loc[mail_data['Category']=='ham', 'Category',]=1


X=mail_data['Message']
Y=mail_data['Category']
#SPLITTING THE DATA INTO TEST AND TRAIN
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

# TRANSFORM THE TEXT DATA INTO VECTORS USING LOGISTIC REGRESSION

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
# print(accuracy_on_test_data)
Actual = Y_train
Predicted = prediction_on_test_data
Actual.head(10)
Predicted.head(10)

confusion_matrix = metrics.confusion_matrix(Actual, Predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


def spamModel(test):
    input_mail = [test]

    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)

    if (prediction[0]==1):
        return [0]
    else:
        return [1]


print(spamModel("hi"))



