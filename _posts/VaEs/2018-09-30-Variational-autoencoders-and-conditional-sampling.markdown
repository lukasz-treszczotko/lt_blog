---
permalink: /machine_learning/cvae/
categories: [Machine Learning, Probability]

---
Variational autoencoders (VAEs) can be very powerful tools when it comes to trying to construct generative models. They have been shown to provide a scalable tool to generate images and sounds. In this note I will show how one can easily sample conditional distributions using a trained variational autoencoder. I will only present the genaral idea. All the detais can be found on my  [GitHub](https://github.com/lukasz-treszczotko/CVAEs) page.

We first train a vanilla VAE on a mixture of Gaussians as in the picture below


![image]({{ site.baseurl }}/assets/cvae1.png)


We use a simple neural network with two hidden layers to train both the encoder and decoder parts. Then we can easily conditionally sample a subset of input variables given that the remaining variables take some predetermined values. To do that we utilize the empirical marginal distribution of the sampled values.

{% highlight python %}
def conditional_sampler(sampled_variables, conditioned_variables,
                        condition_values, num_samples):

    """ sampled variables is a list of indices,
        sampled variables is a list of indices,
        condition_values is a list of values"""

    choice = np.random.choice(len(data[:, 0]), num_samples)
    sampled_data = data[choice, :]
    sampled_data[:, conditioned_variables] = condition_values
    with tf.Session() as sess:
        saver.restore(sess, "./version_" + str(version))
        reconstructed_samples = sess.run(reconstructed_version,
                                         feed_dict={X: sampled_data})
    return reconstructed_samples
{% endhighlight %}

For example, in our simple two-dimensional toy example, conditionally on {% raw %} $$X_0=1$$ {% endraw %} the histogram of values sampled using the trained VAE looks like

![image]({{ site.baseurl }}/assets/cvae2.png)


Sampling the joint distribution gives us the expected picture

![image]({{ site.baseurl }}/assets/cvae3.png)


The upshiot of all of this is as follows: given a set of {% raw %}$$ d $${% endraw %} dependent random variables {% raw %} $$ X_0,\ldots,X_{d-1} $${% endraw %} one can combine variational inference and MCMC to obtain a way of computationally capturing conditional dependencies.






<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
