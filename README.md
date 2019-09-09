# How to
```shell
python optimization.py --activate tf.sigmoid
```
## Low Dimension Experiment
We set n=3, k=2, repeat 500  times; each time trains for 400 epochs.

| activation function    | average loss |
|---------------------- -|--------------|
| none                   | 0.333        |
| tf.sigmoid             | 0.785        |
| tf.tanh                | 0.331        |
| tf.nn.relu             | 0.850        |
| tf.nn.relu6            | 0.828        |

We found:
1. If the activation function is nonnegative, the result is worst even with a constant bias.
1. `tf.tanh` is a little bit better than the case with no activation function.


