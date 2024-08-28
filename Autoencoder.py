def AE():
    import datetime
    import numpy as np
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense

    path_deaths='./deaths_dataset.txt'
    data_deaths=np.loadtxt(path_deaths)

    x_d=[]
    y_d=[]
    for item_d in data_deaths:
        x_d.append(item_d[:39])
        y_d.append(item_d[39])

    x_data_d=np.array(x_d)
    y_data_d=np.array(y_d)



    path='./dataset.txt'
    data=np.loadtxt(path)

    x=[]
    y=[]
    for item in data:
        x.append(item[:39])
        y.append(item[39])

    x_data=np.array(x)
    y_data=np.array(y)



    counter=1
    n_epch=100
    lst_loss=[]
    lst_acc=[]
    lst_predicts=[]
    lst_time=[]

    from sklearn.model_selection import KFold
    kfold = KFold(10,shuffle=True,random_state=0)
    for train, test in kfold.split(x_data,y_data):

        # x_train=x_data[train]
        # x_test=x_data[test]
        # train_labels=y_data[train]
        # test_labels=y_data[test]


        x_train=x_data[train]
        x_test=x_data_d
        train_labels=y_data[train]
        test_labels=y_data_d

        x_train,x_valid,train_labels,valid_labels=train_test_split(x_train,train_labels,test_size=0.2,random_state=0)


        x_train=x_train/np.max(x_train)
        x_test=x_test/np.max(x_test)


        x_train=x_train.reshape((x_train.shape[0],39))
        x_test=x_test.reshape((x_test.shape[0],39))


        model=Sequential()
        model.add(Dense(39,activation='relu',input_shape=(39,)))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(39,activation='sigmoid'))


        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        start=datetime.datetime.now()
        net_history=model.fit(x_train,x_train,epochs=n_epch,validation_split=0.2)
        end=datetime.datetime.now()
        model.save(f'./weights/Autoencoder_fold{counter}.h5')
        trainin_time=end-start
        lst_time.append(trainin_time)

        test_loss,test_acc=model.evaluate(x_test,x_test)
        lst_loss.append(test_loss)
        lst_acc.append(test_acc)

        predicts=model.predict(x_test)
        lst_predicts.append(predicts)
        counter+=1


    acc_avg=str(np.sum(lst_acc)/10)
    loss_avg=str(np.sum(lst_loss)/10)

    acc=str(lst_acc)
    loss=str(lst_loss)



    results_path='./Results_Autoencoder.txt' 
    f1=open(results_path,'a')
    f1.write('Avergae Accuracy: '+acc_avg+'\n'+'Average Loss: '+loss_avg+'\n')
    f1.write('\nAccuracies: '+acc+'\n'+'Losses: '+loss+'\n')
    f1.write('\nTraining Times: \n')
    for j in lst_time:
        f1.write(str(j)+'\n')
    f1.close()


    predicts_path='./Predicts_Autoencoder.txt' 
    f2=open(predicts_path,'a')
    for pred_20 in lst_predicts:
        for pred in pred_20:
            pred=str(pred)
            f2.write(pred+'\n')
        f2.write('\n\n')
    f2.close()


    lst_predicts=np.array(lst_predicts)
    lst_predicts=lst_predicts.reshape(200,39) 
    print(lst_predicts.shape,lst_predicts.ndim)  
    return lst_predicts



lst=AE()




