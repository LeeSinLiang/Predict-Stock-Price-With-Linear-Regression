


# Predict-Stock-Price-With-Linear-Regression

#### This is a Stock Market Prediction using Machine Learning and Linear Regression Model. You can choose whatever CSV Stock File to predict as long they have dates and your target prediction. I recommend downloading historical stock price data at [Yahoo Finance](finance.yahoo.com). Below is a presentation about the whole process of coding this project.

## Table of Contents

 - [Choosing Dataset Wisely](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#choosing-data-set-wisely)
 - [Preprocessing Data](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#preprocessing-data)
 - [Linear Regression Model](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#linear-regression-model)
 - [Training Multiple Models](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#training-multiple-models)
 - [Save Regression Model](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#save-regression-model)
 - [Prediction](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#prediction)
 - [Evaluation](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#evaluation)
 - [Recommended Resources](https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/README.md#recommended-resources)

## Choosing Data Set Wisely

> ***Why do I need a data set?***  
ML depends heavily on data, without data, it is impossible for an “AI” to learn. It is the most crucial aspect that makes algorithm training possible… No matter how great your AI team is or the size of your data set, if your data set is not good enough, your entire AI project will fail! I have seen fantastic projects fail because we didn’t have a good data set despite having the perfect use case and very skilled data scientists.
-- [Towards Data Science](towardsdatascience.com)
####  In conclusion, we must pick dataset that is good for our Linear Regression Model. If I choose AAPL Stocks from 1980 to now...
>![Figure 1: Graph of APPL Stocks from 1980 to 2020][graph]

[graph]: https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/pictures/Figure_1.png "Figure 1: Graph of APPL Stocks from 1980 to 2020"
>Figure 1: APPL Stocks from 1980 to 2020
#### If I try to fit a regression line, the result would be:
>![Figure 2: Graph of APPL Stocks from 1980 to 2020 with regression line][graph1]

[graph1]: https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/pictures/Figure_2.png "Figure 2: Graph of APPL Stocks from 1980 to 2020 with regression line"
#### And if I use r2_score (`from sklearn.metrics import r2_score`) to calculate the r^2 score for our model, I get 0.53 accuracy which is horrible!
#### In the end, I decided to start our model from 2005 to this current year, which is 2020 and fit a regression line to it, and this is the result:

>![Figure 3: Graph of APPL Stocks from 2005 to 2020 with regression line][graph2]

[graph2]: https://github.com/LeeSinLiang/Predict-Stock-Price-With-Linear-Regression/blob/master/pictures/Figure_4.png "Figure 3: Graph of APPL Stocks from 2005 to 2020 with regression line"

#### Accuracy: 0.87 

## Preprocessing Data

> In any Machine Learning process, Data Preprocessing is that step in which the data gets transformed, or _Encoded_, to bring it to such a state that now the machine can easily parse it. In other words, the _features_ of the data can now be easily interpreted by the algorithm.
>  -- [TowardsDataScience](https://towardsdatascience.com/data-preprocessing-concepts-fa946d11c825)

#### There is a lot of preprocessing data techniques , I recommend [this article](https://towardsdatascience.com/data-pre-processing-techniques-you-should-know-8954662716d6) from TowardsDataScience.
#### For this project, I have [impute](https://en.wikipedia.org/wiki/Imputation_(statistics)) NaN(Not a Number) values I saw at the CSV File. We can check whether any of the element is NaN by executing this code: `np.any(np.isnan(mat))` which will then output which of the column(s) have NaN value(s) and remove them: `x[np.isnan(x)] = np.median(x[~np.isnan(x)])`

##  Linear Regression Model
>In statistics, **linear regression** is a **linear** approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple **linear regression**.
>--Wikipedia
#### A simple linear regression equation is `y=mx+b`, whereas `m` is the slope/gradient of the polynomial of the line aka `y`( predict coefficient) and `b` is the intercept of the line (bias coefficient).
>![Equation for m and b](https://wikimedia.org/api/rest_v1/media/math/render/svg/944e96221f03e99dbd57290c328b205b0f04c803)
>##### P/S: Alpha is b , Beta is m
#### Simple linear regression is when one independent variable is used to estimate a dependent variable which is what I use for this project. .
#### When more than one independent variable is present the process is called multiple linear regression。
>The key point in the linear regression is that our dependent value should be continuous and cannot be a discrete value. However, the independent variables can be measured on either a categorical or continuous measurement scale. -- [Machine Learning With Python By IBM](https://www.coursera.org/learn/machine-learning-with-python/)
#### Before we fit our data into the model, we must convert them(date and prices) to numpy arrays `np.asanyarray(dates)` and [reshape](https://stackoverflow.com/questions/39391275/using-x-reshape-on-a-1d-array-in-sklearn) `np.reshape(dates,(len(dates),1))` them as sklearn only accept numpy array or sparse matrix. 
#### After that, we need to [split](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6) our dataset to train data and test data in order to get more accurate evaluation on out of sample(data that didn't train on) accuracy`xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.2)`. I advise to not train and test on the same dataset as it would cause [high variance and low bias](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
#### Now is time for building linear regression model!

    reg = LinearRegression().fit(xtrain, ytrain)
 
 ## Training Multiple Models
The cons of `train_test_split`is that the it's highly dependant on which dataset is trained and tested. One way to approach this problem is to train multiple models and get the highest accuracy model.

    best = 0
    for _ in range(100):
	    xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.2)
	    reg = LinearRegression().fit(xtrain, ytrain)
	    acc = reg.score(xtest, ytest)
	    if acc > best:
	    best = acc
	    
## Save Regression Model

> When dealing with Machine Learning models, it is usually recommended that you store them somewhere. At the private sector, you oftentimes train them and store them before production, while in research and for future model tuning it is a good idea to store them locally. I always use the amazing Python module [pickle](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/) to do so. -- [TowardsDataScience](When%20dealing%20with%20Machine%20Learning%20models,%20it%20is%20usually%20recommended%20that%20you%20store%20them%20somewhere.%20At%20the%20private%20sector,%20you%20oftentimes%20train%20them%20and%20store%20them%20before%20production,%20while%20in%20research%20and%20for%20future%20model%20tuning%20it%20is%20a%20good%20idea%20to%20store%20them%20locally.%20I%20always%20use%20the%20amazing%20Python%20module%20pickle%20to%20do%20so.)

#### We can dump(save) our model to .pickle file using this code:

    with open('prediction.pickle','wb') as f:
	    pickle.dump(reg, f)
	    print(acc)
#### and load it for predictions by using this code:

    pickle_in = open("prediction.pickle", "rb")
    reg = pickle.load(pickle_in)

## Prediction
#### We can predict stock prices by parsing a date integer. For instance, we want to predict the price stock for tomorrow (considering we downloaded dataset today), we can excecute this line of code:

    reg.predict(np.array([[int(len(dates)+1)]]))
## Evaluation
#### There are several evaluation methods, I recommend to read [this article](https://medium.com/@limavallantin/metrics-to-measure-machine-learning-model-performance-e8c963665476)
#### The method I'm going to use is R^2 metric

> As for the R² metric, it measures the **proportion of variability in the target that can be explained using a feature X**. Therefore, assuming a linear relationship, if feature X can explain (predict) the target, then the proportion is high and the R² value will be close to 1. If the opposite is true, the R² value is then closer to 0.
> -- [TowardsDataScience](https://towardsdatascience.com/linear-regression-understanding-the-theory-7e53ac2831b5)
#### As for the formula:
> ![Formula for
> R^2](https://4.bp.blogspot.com/-wG7IbjTfE6k/XGUvqm7TCVI/AAAAAAAAAZU/vpH1kuKTIooKTcVlnm1EVRCXLVZM9cPNgCLcBGAs/s1600/formula-MAE-MSE-RMSE-RSquared.JPG)
> (Sources: [datatechnotes](https://www.datatechnotes.com/2019/02/regression-model-accuracy-mae-mse-rmse.html))
>#### Whereas MSE is Mean Squared Error, MAE is Mean Absolute Error and RMSE is Root Mean Squared Error
#### As for the code to execute r^2 score metrics...

    reg.score(xtest, ytest)
#### Or...

    from sklearn.metrics import r2_score
    r2_score(ytest, reg.predict(xtest))

## Recommended Resources
#### I have compiled a list of resources in the field of AI. Let me know some other great resources on AI, ML, DL, NLP, CV and more by [email](mailto:personcool312@gmail.com)! :)
 - [Machine Learning Mastery](machinelearningmastery.com)
 - [TowardsDataScience](towardsdatascience.com)
 - [Wikipedia](https://www.wikipedia.org)
 - [FreeCodeCamp](https://www.freecodecamp.org)
 - [Machine Learning by Stanford (Great Course)](https://www.coursera.org/specializations/machine-learning)
 - [Machine Learning by IBM](https://www.coursera.org/learn/machine-learning-with-python)
 - [Udacity Intro to Machine Learning](https://www.coursera.org/learn/machine-learning-with-python)
 - [Sentdex](https://www.youtube.com/user/sentdex)
 - [Siraj Raval (Scammer)](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
 - [MIT OpenCourseWare](https://ocw.mit.edu/index.htm)
 - [MIT Deep Learning 6.S191](http://introtodeeplearning.com/)
 - [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
 - [Tensorflow in Practice](https://www.coursera.org/specializations/tensorflow-in-practice)
 - [Scikit Learn](scikit-learn.org)
 - [Tensorflow](www.tensorflow.org)
 - [Pytorch](www.pytorch.org)
 - [Keras](www.keras.org)
 - [Kaggle](www.kaggle.com)

## Thank You!
#### Thanks for spending time on reading this presentation. Hope you like it! Feel free to contact me by [email](mailto:personcool312@gmail.com) :)

