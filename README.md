### environment
1. python2

2. gym, mupen64plus, mupen64plus-bot and deep-learning related packages


### future work
1. currently use a discrete action control, the left/ right control should be spanned into a continuous space.

2. currently use limited training steps for testing the code, drived by the idea of let the agent learn to navigation in the env. but in the real game case **only the total reward of an entire episode makes sense**  ----- to make the agent learn to drive faster.

3. the screen pixels is already resized into 120*160 considering a faster calculation (the original resolution is [480, 640, 3]), but an obvious frame cutton still can be observed through the running window.

4. the validation loop to be added
