from sklearn import tree

#[height, weight, shoe size]
X=[[181,80,44],[177,70,43],[160,60,38],[166,65,40],[181,85,43]]
Y=['male','male','female','female','male']


clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

pred = clf.predict([[190,70,43]])
print pred