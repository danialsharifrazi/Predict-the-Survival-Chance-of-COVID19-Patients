import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv1D,BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report


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
print(x_data.shape,x_data.ndim)
x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],1)


counter=1
n_epch=100
lst_loss=[]
lst_acc=[]
lst_net_histories=[]
lst_reports=[]
lst_AUC=[]
lst_matrix=[]
lst_time=[]
from sklearn.model_selection import KFold
kfold = KFold(10,shuffle=True,random_state=0)
for train, test in kfold.split(x_data,y_data):

    x_train=x_data[train]
    x_test=x_data[test]
    train_labels=y_data[train]
    test_labels=y_data[test]

    x_train=x_train/np.max(x_train)
    x_test=x_test/np.max(x_test)

    x_train,x_valid,train_labels,valid_labels=train_test_split(x_train,train_labels,test_size=0.2,random_state=0)


    from keras.utils import np_utils
    y_train=np_utils.to_categorical(train_labels)
    y_test=np_utils.to_categorical(test_labels)
    y_valid=np_utils.to_categorical(valid_labels)


    from keras.models import Sequential
    from keras.layers import Dense,Flatten,Conv1D,Dropout,BatchNormalization
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy



    model=Sequential()
    model.add(Conv1D(256,3,activation='relu',padding='same',input_shape=(39,1)))
    model.add(Conv1D(256,3,activation='relu'))
    model.add(Conv1D(256,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='sigmoid'))

    
    model.compile(optimizer=Adam(),loss=binary_crossentropy,metrics=['accuracy'])
    start=datetime.datetime.now()
    net_history=model.fit(x_train, y_train,shuffle=True, epochs=n_epch,validation_data=[x_valid,y_valid])
    end=datetime.datetime.now()
    model.save(f'./AE+CNN_fold{counter}.h5')
    trainin_time=end-start
    lst_time.append(trainin_time)

    lst_net_histories.append(net_history)

    test_loss, test_acc=model.evaluate(x_test,y_test)
    lst_loss.append(test_loss)
    lst_acc.append(test_acc)


    predicts=model.predict(x_test)
    predicts=predicts.argmax(axis=1)
    actuals=y_test.argmax(axis=1)

    fpr,tpr,thr=roc_curve(actuals,predicts)
    a=auc(fpr,tpr)
    lst_AUC.append(a)

    r=classification_report(actuals,predicts)
    lst_reports.append(r) 

    c=confusion_matrix(actuals,predicts)
    lst_matrix.append(c)
    counter+=1


acc_avg=str(np.sum(lst_acc)/10)
loss_avg=str(np.sum(lst_loss)/10)
AUC_avg=str(np.sum(lst_AUC)/10)

acc=str(lst_acc)
loss=str(lst_loss)
AUC=str(lst_AUC)



results_path='./results_AE+CNN.txt' 
f1=open(results_path,'a')
f1.write('Avergae Accuracy: '+acc_avg+'\n'+'Average Loss: '+loss_avg+'\n'+'Average AUc: '+AUC_avg+'\n')
f1.write('\nAccuracies: '+acc+'\n'+'Losses: '+loss+'\n'+'AUCs: '+AUC+'\n')
f1.write('\n\nMetrics for all folds: \n\n')
for i in range(len(lst_reports)):
    f1.write(str(lst_reports[i]))
    f1.write('\nTraining Time: '+str(lst_time[i]))
    f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n_______________________\n')
f1.close()