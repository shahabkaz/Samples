# call the bnlearn packageName, also other packages we need to evaluate different structures
library (bnlearn)
library (gRain)
library (gRbase)
library (caTools)

################### Question 1 ########################

# Copy the data set in another one to keep the original
Data1 <- Data

# Getting a sense about the data - if there is null values?
summary(Data1) 

# So we do not have any null value (N/A) in our dataset

# if they are factors or numbers?
str(Data1)

# They are all factors, so they have been discretized before

# We select the v14 (boolean value) as our target value in TAN structure
# 1 - Learning TAN structure
emp.tan <- tree.bayes(Data1, "V14")
graphviz.plot(emp.tan, main = "TAN Employee")

# Also we can use naive bayesian network to learn the structure
# 2 - Learning Naive Bayesian structure
emp.nb <- naive.bayes(Data1, "V14")
graphviz.plot(emp.nb, main = "Naive Employee")

# In TAN structure the nodes can be related to each other

# 3 - Constrained-based algorithms (Grow-Shrink)
emp.gs <- gs(Data1, alpha = 0.05, test = "x2") # alpha = the sig. level
graphviz.plot(emp.gs, main = "Grow Shrink")

# 4 - Score-based algorithms (Hill-Climbing)
emp.hc <- hc(Data1, score = "bic")
graphviz.plot(emp.hc, main = "Hill Climbing")

# In the constrained-based graphs some links are undirected (the software cannot
# establish the direction of the causality). 
undirected.arcs(emp.gs)

# We use Hill-Climbing structure to put undirected arcs into black list
# And also use Grow-Shrink plot to put other undirected arcs into black list
blacklist = data.frame(from = c("V6","V5","V13","V14","V14","V14","V9"), 
                       to   = c("V1","V4","V8" ,"V6" ,"V4" ,"V12","V6"))
# Some of these arcs are rational, so we can decide easily about the deleting arcs in opposite directions:
# Rational arcs: V1 -> V6 : Age -> Marital-status
#                V4 -> V5 : edu level -> edu years
#                V13 -> V8: country -> Race
# about V14, of course we assume other factors are affecting the revenue
#                V9 -> V6 : Sex -> Marital status
# So our other option for Grow-Shrink DAG is:
blacklist2 = data.frame(from = c("V6","V5","V8","V14","V14","V14","V6"), 
                        to   = c("V1","V4","V13","V6","V4" ,"V12","V9"))
blacklist
blacklist2
emp.gsb <- gs(Data1, blacklist = blacklist)
graphviz.plot(emp.gsb, main = "Grow-Shring with blacklist")
emp.gsb2 <- gs(Data1, blacklist = blacklist2)
graphviz.plot(emp.gsb2, main = "Grow-Shring with blacklist2")

# 5 - Bootstrap (using boot.strength function) - learning from data by averaging multiple DAGs
# number of samples: 500
# algorithm: Hill-climbing for learning each sample
# method: bde (we assume 10 instance of our experience, with the same probability for each state)
boot <- boot.strength(Data1, R = 500, algorithm = "hc", 
                      algorithm.args = list(score = "bde", iss = 10))
# The arcs which are stronger than 0.85 and the probability of their direction > 0.5 
boot [(boot$strength > 0.85) & (boot$direction >= 0.5), ]
# The averaged network
emp.boot <- averaged.network (boot, threshold = 0.85)
graphviz.plot(emp.boot, main = "Bootstrap")


# Evaluating the networks (Misclassification Error) using k-fold cross-validation 
# We are going to perform a 5-fold cross validation for all structures
# using the classification error for V14 as a loss function. 
nbcv = bn.cv(Data1, emp.nb, loss = "pred", k = 5, loss.arg = list(target = "V14"))
nbcv   # 0.19
tancv = bn.cv(Data1, emp.tan, loss = "pred", k = 5, loss.arg = list(target = "V14"))
tancv  # 0.17
gscv = bn.cv(Data1, emp.gsb, loss = "pred", k = 5, loss.arg = list(target = "V14"))
gscv   # 0.1829862
gscv2 = bn.cv(Data1, emp.gsb2, loss = "pred", k = 5, loss.arg = list(target = "V14"))
gscv2   # 0.1827208
hccv = bn.cv(Data1, emp.hc, loss = "pred", k = 5, loss.arg = list(target = "V14"))
hccv   # 0.24
bootcv = bn.cv(Data1, emp.boot, loss = "pred", k = 5, loss.arg = list(target = "V14"))
bootcv # 0.21

# TAN structure has the best prediction performance in terms of predicting the 
# target variable and misclassification rate. After that the Grow-Shrink 
# structures (Second is a little better) and Naive are the best
# So we can conclude that classification algorithms perform well in prediction of
# target and having low values in misclassification rates 

# Evaluating the networks (Prediction Performance) using AUC and ROC curve
# We use predict function for the best cross-validation structure as well as 
# all structures we have (Naive, TAN, GS, HC, Bootstrap)

# TAN
netcvfit1 = as.grain(tancv[[1]]$fitted)
emp_test1 = Data1[tancv[[1]]$test, ]
pred_test1 = predict(netcvfit1, response = c("V14"), newdata = emp_test1, 
                     predictors = names(emp_test1)[-14], type = "distribution")

# Here we encountered to error smooth argument and exit
# We continue with other networks:


# Copy the data set in another one to keep the original
Data2 <- Datatest

# Getting a sense about the data - if there is null values?
summary(Data2) 

# So we do not have any null value (N/A) in our dataset

# if they are factors or numbers?
str(Data2)

# Evaluating the networks (Prediction Performance) using AUC and ROC curve (Cont.)
# This time we use the Data2 as a test dataset

# AUC and ROC curve for Naive
fitted_nb = bn.fit(emp.nb, Data1)
nb_fit <- as.grain(fitted_nb)
pred_nb = predict(nb_fit, response = c("V14"), newdata = Data2, 
                    predictors = names(Data2)[-14], type = "distribution")
colAUC(pred_nb$pred$V14, Data2[, 14], plotROC = TRUE)
#                    <=50K      >50K
# <=50K vs. >50K 0.8710928 0.8710928

# AUC and ROC curve for TAN
fitted_tan = bn.fit(emp.tan, Data1)
tan_fit <- as.grain(fitted_tan)
pred_tan = predict(tan_fit, response = c("V14"), newdata = Data2, 
                  predictors = names(Data2)[-14], type = "distribution")
colAUC(pred_tan$pred$V14, Data2[, 14], plotROC = TRUE)
#                    <=50K      >50K
# <=50K vs. >50K 0.8880065 0.8880065

# AUC and ROC curve for GS
fitted_gs = bn.fit(emp.gsb, Data1)
gs_fit <- as.grain(fitted_gs)
pred_gs = predict(gs_fit, response = c("V14"), newdata = Data2, 
                   predictors = names(Data2)[-14], type = "distribution")
colAUC(pred_gs$pred$V14, Data2[, 14], plotROC = TRUE)
#                    <=50K      >50K
# <=50K vs. >50K 0.8681308 0.8697158

# AUC and ROC curve for GS
fitted_gs2 = bn.fit(emp.gsb2, Data1)
gs_fit2 <- as.grain(fitted_gs2)
pred_gs2 = predict(gs_fit2, response = c("V14"), newdata = Data2, 
                  predictors = names(Data2)[-14], type = "distribution")
colAUC(pred_gs2$pred$V14, Data2[, 14], plotROC = TRUE)
#                    <=50K      >50K
# <=50K vs. >50K 0.8677789 0.8677789

# AUC and ROC curve for HC
fitted_hc = bn.fit(emp.hc, Data1)
hc_fit <- as.grain(fitted_hc)
pred_hc = predict(hc_fit, response = c("V14"), newdata = Data2, 
                  predictors = names(Data2)[-14], type = "distribution")
colAUC(pred_hc$pred$V14, Data2[, 14], plotROC = TRUE)
#                    <=50K      >50K
# <=50K vs. >50K 0.8769131 0.8769433

# Note: We cannot use "grain" function to transform a bn which is built based on 
# cpt to a junction tree

# TAN structure has the best prediction performance in terms of AUC and ROC curve
# (Just like Cross-validation). In fact the percentage of correct prediction
# in TAN (for both categories: <= 50K and > 50K) is more than other BNs. After 
# that the Hill-Climbing and Naive are the best with Data2 dataset as our test set.

################### Question 2 ########################

# comparison between arcs (Naive)
arc.strength(emp.nb, data = Data1, criterion = "x2")

# comparison between arcs (TAN)
arc.strength(emp.tan, data = Data1, criterion = "x2")

# The applications of different structures are different:
# We use special purpose DAGs (Naive and TAN) to predict class membership 
# probabilities based on the common aspects, so in comparison between TAN and 
# Naive Bayesian DAGs, we see TAN is better (in structure), because it shows more 
# relationships

arc.strength(emp.gsb, data = Data1, criterion = "x2") # all arcs are strong enough - useful to discover casualities
arc.strength(emp.hc, data = Data1, criterion = "x2")  # all arcs are strong enough - useful for prediction
arc.strength(emp.boot, data = Data1, criterion = "x2") # all arcs are strong enough - averaging multiple DAGs

# Influence Analysis: Now we can test the nodes we have found from the 
# Sensitivity analysis: we use TAN structure because it had the best performance in prediction the target variable
fitted <- bn.fit(emp.tan, Data1, method = "mle")
fitted

fitted$V12
fitted$V2
fitted$V14

str(Data1)

# We use cpquery function to get the approximate inference and see how the target variable has sensitivity to the selected variables

# Because we do not have enough time, we are going to select a sample and caannot see all joint and conditional probabilities for V14
# Also we have selected V12 for our sample

cpquery(fitted, event = (V14 == "<=50K"), evidence = (V12 == "(33.7,66.3]"))
cpquery(fitted, event = (V14 == "<=50K"), evidence = (V12 == "(66.3,99.1]"))
cpquery(fitted, event = (V14 == ">50K"), evidence = (V12 == "(33.7,66.3]"))
cpquery(fitted, event = (V14 == ">50K"), evidence = (V12 == "(66.3,99.1]"))

# As we can see changes in states of V12 (working hours), has more effect on revenues more than $50K (9%) but it has less effect on revenues less than $50K (3%)

# We can test other variables and states and see which of them has the most effect on V14
# Also we can use combination of variables to see the effects of changing them on V14

################### Question 3 ########################

# We have seen the approximate inference in question 2 (cpquery)
# Now we can test Exact inference with querygrain command (function)

# Transform the bn into a junction tree (using "as.grain" function) and compute
# probability tables ("compile" function)
# In this example we use Hill-Climbing structure
bn <- bn.fit(emp.hc, data = Data1)
# Transform the bn into a junction tree (using "as.grain" function) and compute
# probability tables ("compile" function)
junction <- compile(as.grain(bn))
junction

querygrain(junction, nodes = "V14")$V14 # Exact Inference
# This shows that about 75% of employees (our sample) have the revenue of $50K per year and 25% of them have more than $50K

# Let see how changing the states of other variables can affect this percentages
# for example sex
str(Data1)
jsex <- setEvidence(junction, nodes = "V9", states = "Female")
querygrain(jsex, nodes = "V14")$V14 # Show the probability

# We can see if employee is a woman, the percentage of getting more than $50K per year can decreased to just 11%
# In other words just 11% of all woman of our sample get more than $50K each year
# Let try with men
jsex <- setEvidence(junction, nodes = "V9", states = "Male")
querygrain(jsex, nodes = "V14")$V14 # Show the probability
# We see we have an increase of 6% (from 25% to 31%) for receiving more than $50K if the sample is a man

# The default value displayed in the querygrain is marginal probability 
# distribution. but we can also display the joint or conditional. 
# e.g. P(V14, V9 | V13 = Canada)
jcon <- setEvidence(junction, nodes = "V13", states = "Canada")
SxT.cpt <- querygrain(jcon, nodes = c("V14", "V9"), type = "joint")
SxT.cpt

# This shows in Canada how the percentages are distributed

# Also we can use conditional probabilities
SxT.cpt <- querygrain(jcon, nodes = c("V14", "V9"), type = "conditional")
SxT.cpt
