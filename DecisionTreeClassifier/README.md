from sklearn import tree   

#[height,weight,shoe-size]

x = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[167,61,40],
     [166,65,44],[190,90,47],[175,64,39],[177,70,44],[159,55,38],[171,75,42],[181,85,43]]

y = ['male','male','female','female','female','female','male',
     'male','male','female','male','male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

prediction = clf.predict([[160,60,40]])
print(prediction)
