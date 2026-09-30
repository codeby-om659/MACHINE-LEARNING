from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
y_true=[1,0,1,1,0,1,0]#what is actually happened
y_prediction=[1,0,1,0,0,1,1]#what is guess
print("accuracy:",accuracy_score(y_true,y_prediction))
print("precision:",precision_score(y_true,y_prediction))
print("recall:",recall_score(y_true,y_prediction))
print("f1_score:",f1_score(y_true,y_prediction))
