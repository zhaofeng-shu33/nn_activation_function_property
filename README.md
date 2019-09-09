# How to
```shell
python optimization.py --activate tf.sigmoid
```
## Low Dimension Experiment
We set n=3, k=2, repeat 100  times; each time trains for 100 epochs.

| activation function | average loss |
|---------------------|--------------|
| none                | 0.142        |
| sigmoid             | 0.219        |
| tanh                | 0.343        |
| relu                | 0.270        |
| relu6               | 0.280        |
