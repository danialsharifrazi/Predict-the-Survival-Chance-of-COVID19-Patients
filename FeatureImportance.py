import numpy as np


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


    x_data,y_data=ReadData_SingleModel()
    # x_data,y_data=ReadData_MixedWithAEModel()



    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(n_estimators = 10)
    model.fit(x_data,y_data)
    print(model.feature_importances_)

    path=f'./feature_importances.txt'
    f2=open(path,'a')
    f2.write(str(model.feature_importances_))
    f2.close()

      
RF_model()
