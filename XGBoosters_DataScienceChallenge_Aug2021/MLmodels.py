
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imxgboost.imbalance_xgb import imbalance_xgboost




def cross_val(Classifier, x_train, y_train):
    skf = StratifiedKFold(n_splits=5)
    accuracies = cross_val_score(estimator=Classifier, X=x_train, y=y_train, cv=skf)
    print(accuracies)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

def conf_mat(model, x_val, y_val):
    y_pred = model.predict(x_val)

    # --classification report --
    print('\nClassification Report:')
    print(metrics.classification_report(y_val, y_pred, labels=[0,1]))

    # --confusion matrix
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_val, y_pred))
    print('---------------------------------------------------------------------------------------')

class MachineLearningModels:

    def ml_model(x_train, y_train, x_val, y_val):
        
        print('Logistic Regression: \n')
        LR = LogisticRegression(random_state = 0,solver='liblinear')
        LR.fit(x_train, y_train)
        cross_val(LR, x_train, y_train)
        conf_mat(LR, x_val, y_val)
        
        print('Gaussian Naive Bayes: \n')
        GNB = GaussianNB()
        GNB.fit(x_train, y_train)
        cross_val(GNB, x_train, y_train)
        conf_mat(GNB, x_val, y_val)
        
        print('KNeighbors Classifier: \n')  
        KNC = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        KNC.fit(x_train, y_train)
        cross_val(KNC, x_train, y_train)
        conf_mat(KNC, x_val, y_val)
        
        print('SGD Classifier: \n')
        sgdc = SGDClassifier(max_iter=1000, tol=0.01)
        sgdc.fit(x_train, y_train)
        cross_val(sgdc, x_train, y_train)
        conf_mat(sgdc, x_val, y_val)
        
    #     print('Perceptron Classifier: \n') 
    #     Perceptron = Perceptron()
    #     Perceptron.fit(x_train, y_train)
    #     cross_val(Perceptron, x_train, y_train)
    #     conf_mat(Perceptron, x_val, y_val)
        
        print('Ridge Classifier: \n')
        RC = RidgeClassifier()
        RC.fit(x_train, y_train)
        cross_val(RC, x_train, y_train)
        conf_mat(RC, x_val, y_val)
        
        print('Decision Tree Classifier: \n')
        DT = DecisionTreeClassifier()
        DT.fit(x_train, y_train)
        cross_val(DT, x_train, y_train)
        conf_mat(DT, x_val, y_val)
        
        print('Random Forest Classifier: \n')
        RFC = RandomForestClassifier(n_estimators = 10, criterion = "gini", random_state = 0)
        RFC.fit(x_train, y_train)
        cross_val(RFC, x_train, y_train)
        conf_mat(RFC, x_val, y_val)
        
        print('Ada Boost Classifier: \n')
        ABC = AdaBoostClassifier(n_estimators=100)
        ABC.fit(x_train,y_train)
        cross_val(ABC, x_train, y_train)
        conf_mat(ABC, x_val, y_val)
        
        print('LightGBM Classifier : \n')
        lgb = LGBMClassifier()
        lgb.fit(x_train,y_train)
        cross_val(lgb, x_train, y_train)
        conf_mat(lgb, x_val, y_val)
        
        print('XGBoost Classifier: \n')
        XGB = XGBClassifier()
        XGB.fit(x_train,y_train)
        cross_val(XGB, x_train, y_train)
        conf_mat(XGB, x_val, y_val)
        
#         print('SVC-linear Classifier: \n')
#         SVCl = SVC(kernel = 'linear', random_state = 0)
#         SVCl.fit(x_train, y_train)
#         cross_val(SVCl, x_train, y_train)
#         conf_mat(SVCl, x_val, y_val)
        
#         print('SVC-rbf Classifier: \n')
#         SVC = SVC(kernel = 'rbf', random_state = 0)
#         SVC.fit(x_train, y_train)
#         cross_val(SVC, x_train, y_train)
#         conf_mat(SVC, x_val, y_val)
        
        print('Imbalance_XGBoost: \n')
        imb_xgb = imbalance_xgboost(special_objective='focal',focal_gamma=1.3)
        imb_xgb.fit(x_train.values, y_train)
        cross_val(imb_xgb, x_train.values, y_train)
          
        y_preds = imb_xgb.predict_determine(x_val.values)
        # --classification report --
        print('Classification Report')
        print(metrics.classification_report(y_val, y_preds, labels=[0,1]))
        
        # --confusion matrix
        print('Confusion Matrix:')
        print(confusion_matrix(y_val, y_preds))



    
