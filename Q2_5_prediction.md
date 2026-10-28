# Q2.5 prediction

Figure 1(a) shows movement in every direction, so mirrored and rotated directions should have some representation in the training sample.

**Will model A pass the mirror test?**

I expect some difficulty with mirroring because the digit appearance changes, even though the mirrored motion directions appear in training.

**Will model A pass the rotation test?**

I expect rotation to be harder than mirroring because it changes digit appearance more, despite the rotated directions appearing in training.

**At what speed should model A break down, given the reach estimate in Q2.3 and Figure 1?**

I expect difficulty at 3×: typical movement is about 9.96 pixels between frames, exceeding model A's estimated reach of 8 pixels per time step.

