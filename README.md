# image-denoising-mrf

This simple program demonstrates how a Markov network can be applied to the problem of image denoising.

```bash
# To run with default values of parameters w_e and w_s, execute
python denoise.py input.png

# Alternately, pass the values of w_e and w_s as second and third arguments respectively:
python denoise.py input.png 8 10
```

See https://blog.statsbot.co/probabilistic-graphical-models-tutorial-d855ba0107d1 for the full blog post explaining the example.
