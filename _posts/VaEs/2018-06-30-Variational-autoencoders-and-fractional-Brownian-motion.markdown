---
permalink: /machine_learning/vae/
categories: Probability, Machine Learning

---

A well known fact about the fractional Brownian motion (fBm for short)
is that it is not a semimartingale and in particular it cannot be written as a stochastic
integral with respect to a Brownian motion and, at the same time, be
adapted to this Brownian motion. However, there is a representation of
fractional Brownian motion (see for example Chapter 6 in [Pipiras and Taqquu](https://www.cambridge.org/core/books/longrange-dependence-and-selfsimilarity/EC0867FA235989C077341B00822BF829)), which says that for any {% raw %}
$$a>0$$ {% endraw %} and {% raw %} $$ (W_s) $$ {% endraw %} being a
Brownian motion on {% raw %} $$ [0,a] $$ {% endraw %}, there is a
deterministic function {% raw %} $$ f(t,u,a,H) $$ {% endraw %} such that

{% raw %}

$$B^H_t:=int_0^a f(t,u,a,H)dW_u$$

{% endraw %}

is a fractional Brownian motion with Hurst coefficient {% raw %}
$$ H $$ {% endraw %}. This suggests that, at list in principle Variational
Autoencoders can be used to provide a generative model of fBm. The idea is to
to encode the paths of the stochastic process (in this case - fBM) onto the
latant variable space that is a discretized version of a standard Brownian
motion. In the simple setting here, let a neural network with two hidden
layers learn the latent representation.




<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
