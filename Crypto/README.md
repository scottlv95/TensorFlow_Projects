Learning Tensorflow with Crypto

This model predicts whether a specified cryptocurrency will rise or fall base on features of other cryptocurrency.

Below is the list of cryptocurrency I am currently using.
"BTC-USD","LTC-USD","ETH-USD","BCH-USD"

Currently using LSTM layers, as it is a time sensitive data thus using recurrent neural network makes more sense
(CUDnn version used automatically with "tanh" set as activation function)
3 layers of LSTM followed by 1 Dense layer and an output layer.

Just running 10 epochs to predict LTC_USD as gives an average validation-accuracy of around 58%
