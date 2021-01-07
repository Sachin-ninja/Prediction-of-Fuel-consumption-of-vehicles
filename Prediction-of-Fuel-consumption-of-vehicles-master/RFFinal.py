# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk    
    
        
def getfile():
        global dataset
        dataset=dataset=pd.read_csv(r'C:\Users\Mouni\Desktop\Machine Learning\Group Project\Fuel_Consumption_2014-19.csv')
        mp.defmodel()


class manipulate:
    @staticmethod
    def defmodel():
        global X_train,X_test,y_train,y_test
        X=dataset.iloc[:,2:5].values
        y=dataset.iloc[:,5].values
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        X_train=sc.fit_transform(X_train)
        X_test=sc.fit_transform(X_test)
        import statsmodels.api as sm
        summar=sm.OLS(y,X).fit()
        print(summar.summary())
    
    @staticmethod
    def savemodel():
        global y_pred,model_from_pickle
        regressor=RandomForestRegressor()
        regressor.fit(X_train,y_train)
        y_pred=regressor.predict(X_test)
        import pickle
        saved_model=pickle.dumps(regressor)
        model_from_pickle=pickle.loads(saved_model)
        
        #(print(model_from_pickle.predict([[float(x),int(y),int(z)]]))
    
    @staticmethod
    def predicterror():
        from sklearn import metrics
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
        print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
        print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
        
        
    @staticmethod  
    def makegui():
        global ent1,ent2,ent3
        lb1=tk.Label(frame,text="Enter the Engine_Size:",font="Times 12")
        lb1.grid(row=0,column=15)
        lb2=tk.Label(frame,text='Enter CO2 emissions in gm/cc:',font="Times 12")
        lb2.grid(row=1,column=15)
        lb3=tk.Label(frame,text='Enter No. of Cyliner:',font="Times 12")
        lb3.grid(row=2,column=15)
        ent1=tk.Entry(frame)
        ent1.grid(row=0,column=16)
        ent2=tk.Entry(frame)
        ent2.grid(row=1,column=16)
        ent3=tk.Entry(frame)
        ent3.grid(row=2,column=16)
        bt=tk.Button(frame,command=mp.getvalues,text='Predict')
        bt.grid(row=3,column=15)           
        frame.grid()
    
    @staticmethod
    def getvalues():
        #x1=int(ent1.get())
        #y=(ent2.get())
        #z=(ent3.get())
        '''import pickle
        saved_model=pickle.dumps(regressor)
        model_from_pickle=pickle.loads(saved_model)'''
        global x
        x=model_from_pickle.predict([[float(str(ent1.get())),int(str(ent2.get())),int(str(ent3.get()))]])/100
        #lb4=tk.Label(frame,text=model_from_pickle.predict([[float(ent1.get()),int(ent2.get()),int(ent3.get())]]))
        global ent4
        lb4=tk.Label(frame,text=x)
        lb4.grid(row=4,column=15)
        lb5=tk.Label(frame,text="Enter No of Km's you to travel in your car: ",font="TImes 12")
        lb5.grid(row=5,column=15)
        ent4=tk.Entry(frame)
        ent4.grid(row=5,column=16)
        bt2=tk.Button(frame,text='Calculate',command=mp.getkm)
        bt2.grid(row=6,column=15)
        #lb6=tk.Label(frame,text=yv*x)
        #lb6.grid(row=7,column=15)
        
    @staticmethod
    def getkm():
        yv=ent4.get()
        lb6=tk.Label(frame,text=(x*int(yv)))
        lb6.grid(row=7,column=15)
        lb7=tk.Label(frame,text='Devoped by:',font="Times 14 bold")
        lb7.grid(row=40,column=40)
        lb8=tk.Label(frame,text='-S.SACHIN GOUD',font='Times 12')
        
        
root=tk.Tk()
frame=tk.Frame(root,width=500,height=1000)
mp=manipulate()
getfile()
mp.defmodel()
mp.savemodel()
mp.predicterror()
mp.makegui()
root.mainloop()





