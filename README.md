# Numerai
Tensorflow wide and deep model and xgboost


Numerai is a new kind of hedge fund that pays out a reward to data scientists that can improve their meta model.
The idea is to take a large number of uncorrelated high performing models and ensemble them.
Money is divied out based on leaderboard ranking awarding the top 100.

I tried out an xgboost model a wide and deep model using tensorflow.

Wide and Deep Learning research paper:
https://arxiv.org/abs/1606.07792

The xgboost model was rather poor but the tensorflow model ended up scoring the same logloss as the top 
performing models.
