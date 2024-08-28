import datetime
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score


def ReadData_SingleModel():
    path='./dataset.txt'
    data=np.loadtxt(path)

    x=[]
    y=[]
    for item in data:
        x.append(item[:39])
        y.append(item[39])

    x_data=np.array(x)
    y_data=np.array(y)
    return x_data,y_data


def ReadData_MixedWithAEModel():

    path='./dataset.txt'
    data=np.loadtxt(path)

    x=[]
    y=[]
    for item in data:
        x.append(item[:39])
        y.append(item[39])


    import Autoencoder
    reconstracted=Autoencoder.AE()
    x_d=[]
    y_d=[]
    for item_d in reconstracted:
        x_d.append(item_d[:39])
        y_d.append(1)
    x.extend(x_d)
    y.extend(y_d)


    x_data=np.array(x)
    y_data=np.array(y)
    x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],1)
    return x_data,y_data


def RF_model():


    # x_data,y_data=ReadData_SingleModel()
    x_data,y_data=ReadData_MixedWithAEModel()


    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(n_estimators = 10)


    counter=1
    lst_acc=[]
    lst_time=[]
    lst_AUC=[]
    lst_matrix=[]
    lst_reports=[]

    from sklearn.model_selection import KFold
    kfold = KFold(10,shuffle=True,random_state=0)
    for train, test in kfold.split(x_data,y_data):

        x_train=x_data[train]
        x_test=x_data[test]
        y_train=y_data[train]
        y_test=y_data[test]


        x_train=x_train/np.max(x_train)
        x_test=x_test/np.max(x_test)


        x_train=x_train.reshape((x_train.shape[0],39))
        x_test=x_test.reshape((x_test.shape[0],39))


        start=datetime.datetime.now()
        model.fit(x_train,y_train)
        end=datetime.datetime.now()
        trainin_time=end-start
        lst_time.append(trainin_time)

        predicts=model.predict(x_test)
        actuals=y_test

        acc=accuracy_score(actuals,predicts)
        lst_acc.append(acc)

        r=classification_report(actuals,predicts)
        lst_reports.append(r)

        c=confusion_matrix(actuals,predicts)
        lst_matrix.append(c)

        fpr,tpr,thr=roc_curve(actuals,predicts)
        a=auc(fpr,tpr)
        lst_AUC.append(a)


        path=f'./predicts_RF_fold{counter}.txt'
        f2=open(path,'a')
        f2.write(str(predicts))
        f2.close()



        path=f'./RF_fold{counter}.txt'
        f1=open(path,'a')
        f1.write('Accuracy: '+str(acc)+'\nConfusion Matrix: \n'+str(c)+'\n\nAUC: '+str(a)+'\nTraining Times:'+str(trainin_time)+'\n\n'+str(r))
        f1.close()
        counter+=1

      
RF_model()
