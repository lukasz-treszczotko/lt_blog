---
permalink: /machine_learning/vae/
categories: [Machine Learning, Probability]

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

$$B^H_t:=\int_0^a f(t,u,a,H)dW_u$$

{% endraw %}

is a fractional Brownian motion with Hurst coefficient {% raw %}
$$ H $$ {% endraw %}. This suggests that, at least in principle Variational
Autoencoders can be used to provide a generative model of fBm. The idea is to
to encode the paths of the stochastic process (in this case - fBM) onto the
latant variable space that is a discretized version of a standard Brownian
motion. In the simple setting here, let a neural network with two hidden
layers learn the latent representation.

The encoder-decoder structure we use is relatively simple. The encoder is
standard.

{% highlight python %}
def make_encoder(data, code_size, scope='encoder'):
    with tf.name_scope('encoder'):
        h_1 = tf.layers.dense(data, hidden_size, activation=tf.nn.relu, name='hidden_1')
        h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.relu, name='hidden_2')
        loc = tf.layers.dense(h_2, code_size, name='loc')
        scale = tf.layers.dense(h_2, code_size, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)
{% endhighlight %}

The prior (latent structure) is given by a standard multivariate normal
distribution.

{% highlight python %}
def make_prior(code_size):
    with tf.name_scope('prior'):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        return tfd.MultivariateNormalDiag(loc, scale)
{% endhighlight %}

The decoder is again a neural network which learns both mean and diagonal
variance (it appears to do much worse if diagonal variance is set to
a multiple of some hyperparameter).

{% highlight python %}
def make_decoder(code, data_shape):
      with tf.name_scope('decoder'):

          h_1 = tf.layers.dense(code, hidden_size, tf.nn.relu, name='hidden_1')
          h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.relu, name='hidden_2')

          loc = tf.layers.dense(h_2, np.prod(data_shape), name='loc')
          scale = tf.layers.dense(h_2, np.prod(data_shape),
                                  activation=tf.nn.sigmoid,
                                  name='scale',
                                  use_bias=False)
          return tfd.MultivariateNormalDiag(loc=loc,
                                            scale_diag=scale)
{% endhighlight %}

The loss we are trying to minimize is the standard -ELBO. After training we
can generate samples very easily.

{% highlight python %}
def sample_generator():
    with tf.Session() as sess:

        code_sample = sess.run(prior.sample())
        saver.restore(sess, "./version_" + str(version))
        code_sample = np.expand_dims(code_sample, axis=0)
        generated_sample = sess.run(reconstructed_version,
                                    feed_dict={code: code_sample})
    return np.squeeze(generated_sample)
{% endhighlight %}

Whole code can be seen [here](https://github.com/lukasz-treszczotko/GNE/blob/master/fbm_sota.py).
Here are sample paths generated for Hurst coefficient {% raw %} $$H=0.7$$
{% endraw %} (after a considerable fine-tuning of the
parameters and at the same time computing log probability on the testing
set to prevent overfitting).

![image]({{ site.baseurl }}/assets/v1.png)

![image]({{ site.baseurl }}/assets/v2.png)

![image]({{ site.baseurl }}/assets/v3.png)

The overall shape seems sensible, although the rough nature of paths of fBm is
not recovered, which is not surprising, given that in general VAEs seem to
make everything a bit "blurry".




<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
