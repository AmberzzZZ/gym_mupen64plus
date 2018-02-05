### environment
1. python2

2. gym, mupen64plus, mupen64plus-bot and deep-learning related packages

### how to run it
1. run the 'train.py' file to start training, the network weights will be saved in data/model_weights
```
python train.py
```

2. run the 'run_weights.py' file to test the network performance.
```
python run_weights.py
```

3. run the 'visualization.py' file to visualize the output of the first convolution network
```
python visualization.py
```

### future work
1. currently use a discrete action control, the left/ right control should be spanned into a continuous space.

2. currently use limited training steps for testing the code, drived by the idea of let the agent learn to navigation in the env. but in the real game case **only the total reward of an entire episode makes sense**  ----- to make the agent learn to drive faster.

3. the screen pixels is already resized into 120*160 considering a faster calculation (the original resolution is [480, 640, 3]), but an obvious frame cutton still can be observed through the running window.

4. the validation loop to be added

5. visualization

6. refer to the stanford paper, add an offline search agent to generate dataset first. Then train the CNN to imitate the offline agent.






