<center>
<p> "<h1># heart-disease-prediction</h1></p>
</center>
<h3>this project in artificial intelligence that help you to predict weather you have aheart disease or not</h3>
1) Preprocessing:<br>
First deal with nulls ,I replace null cells with mean of columns if columns are numerical <br>
and with mode of columns if columns are strings ,drop nulls not preferred because data is already small.<br>
Second deal with string columns (features)by get dummies(one hot encoder) and label by replace <br>
function or label encoder.<br>
Third check if there is duplicate row and remove.<br>
Fourth check outliers and index of rows that have outliers and drop them to reduce skewness of <br>
features.<br>
Fifth split dataset to features and labels<br>
Last step in preprocessing feature scaling on features is important to make gradient descent more faster <br>
and make all columns have same range of values.<br>
2) Feature selection<br>
It is an important step in order to drop any column that has low correlation between it and label (less than <br>
0.1) , reduce number of features is an important step to increase accuracy.<br>
3) Train and Test model<br>
1. Split dataset into data in order to train model and small portion of data (between 20 % and 30%) <br>
to test model.<br>
2. Random state to controls shuffling applied to data before applying the split. <br>
4) Used algorithms <br>
Logistic <br>
Regression<br>
SVM Decision Tree KNN Random <br>
Forrest<br>
XGbosst<br>
1)Logistic Regression<br>
Logistic Regression is used for predicting categorical data <br>
Sigmoid function is a function of linear Regression equation (w*x +b) , predict probability of Heart disease <br>
is exist ,if probability less than 0.5 then heart disease is not exist and if more than 0.5 (threshold) then <br>
heart disease is exist but label is 0(not exist) or 1(exist).<br>
Hyperparameters <br>
1) Solver: used to optimize data ex. lbfgs(best value in our model) deals with small dataset to make <br>
output more accurate.<br>
2) C: regularization hyperparameter is used to decrease overfitting range(between 0.5 and 1 to avoid <br>
overfitting) , the most suitable value in our model is 1(1 is the best value in our model)<br>
2)SVM<br>
Support vector machine is a type of classification algorithm that used to classifies data according <br>
to the features.<br>
It separates classes by linear kernal ,polynomial ,non linear according to data to predict what <br>
class the predicted value belong to ex. we have two classes (have heart disease , does not have <br>
heart disease) it separates between two classes and determine whether the predicted value <br>
belong to class 1 or class 2 and put the point in class it belongs to.<br>
It should be with large margins and equidistance between support vectors from two classes<br>
Hyperparameters<br>
1) Kernal: it depends on data on scatter plot ,values: linear, poly, rbf(value in our model is <br>
poly to separate between classes)<br>
2) C:Regularization parameter default is 10 increasing it reduces error but may leads to <br>
overfitting(value in our model is (value in our model is 3) <br>
3)Decision Tree<br>
Is used in classification is select features with the highest information gain to be splitted (root <br>
node) ,if this split have entropy more than zero (impurity) either in left branch or right branch, <br>
do another split till reach max depth or zero entropy.<br>
Hyperparameters<br>
1. Max depth:it should not be very large value in order not to increase complexity of <br>
algorithm and not to increase error (value in our model is 4)<br>
2.Max _feature =consider number of feature (not all features) to calculate information gain <br>
of it in order to choose one of them to be splitted .(value in our model 3)<br>
3.criterion=measure impurity of the split of branch ,value=entropy, gini. (value in our model <br>
is entropy)<br>
4)knn<br>
it is an classification algorithm ,it predict by calculate distance between point and its nearest <br>
neighbors by ecludian or matahan .<br>
hyperparameters<br>
1)nearest neighbors=number of nearest neighbor.(best value in our model=4)<br>
2)metric=value(minkowski) with p=1 using Manhattan to calculate distance(this our best value for our <br>
model) rather than p=2(eucludian).<br>
 
5)Random forest:<br>
Is used in classification is select features with the highest information gain to be splitted (root <br>
node) ,if this split have entropy more than zero (impurity) either in left branch or right branch, <br>
do another split till reach max depth or zero entropy. It is better than decision because it <br>
consists of many trees by sample with replacement from original training set to build more 
trees.<br>
Hyperparameters: <br>
1.Max depth:it should not be very large value in order not to increase complexity of algorithm <br>
and not to increase error (value in our model is 4)<br>
2.Max _feature =either log2 or sqrt , consider number of feature (not all features) to calculate <br>
information gain of it in order to choose one of them to be splitted .(value in our model is <br>
sqrt)
3.criterion=measure impurity of the split of branch ,value=entropy, gini.(value in our model is <br>
entropy)<br>
4.number of trees= value in our model(100)<br>
6)Xgboost:<br>
• XGBoost stands for Extreme Gradient Boosting.<br>
• It is a tree boosting algorithm that can be used for both classification and regression tasks.<br>
• XGBoost is a popular choice for machine learning competitions because it is very efficient <br>
and can achieve state-of-the-art results.<br>
• XGBoost has many hyperparameters that can be tuned to improve the performance of the <br>
model.<br>
• Some of the most important hyperparameters include the learning rate, the number of <br>
trees, the maximum depth of each tree, and the regularization parameters.
• The optimal values for these parameters will vary depending on the dataset and the task <br>
at hand. It is important to experiment with different values to find the best results.
<h1>Final conclusion</h1> <br>
Logistic Svm Decision tree Knn Random forest Xg boost<br>
Test acc=0.849 Test acc=0.864 Test acc=0.83 Test acc =0.849 Test acc=0.886 Test acc=0.862<br>
Train acc=0.84 Train=0.878 Train <br>
acc=0.849<br><br>
Train <br>
acc=0.849<br>
Train <br>
acc=0.849<br>
Train <br>
acc=0.848<br>
Finally<br>
This accuracy in each model is the best we reach ,sometimes we have very high test acc but train <br>
acc less than high by more than 4% which is not good so we should balance between them to <br>
avoid underfitting or overfitting.<br>
