---
permalink: /probability/hawkes/

---

You can view the preprint here [get the PDF]({{ site.url }}/assets/Hawkes.pdf).
Simulating paths of the intensity process in Python is relatively straightforward.
Let us set some hyperparameters first.

{% highlight python %}
lambda_1 = 0.3
lambda_0 = 0.31
T_0 = 1
T = 10000.
time_limit = 140.
{% endhighlight %}

We define some helpful functions. The first one {% raw %}
$$a(r)$$ {% endraw %} computes the instability parameter, i.e., how close to 'explosion'
we are.

{% highlight python %}
def a(r):
    return  1 - lambda_1/r
{% endhighlight %}

Next we provide a funcion creating a birth event, which returns a list [birth date,
offspring size, offspring lifespan].

{% highlight python %}
def make_birth(t):
    w = np.random.exponential(scale=[1,1])
    return [t, w[0], w[1]]
{% endhighlight %}

Now, a function to compute waiting times.

{% highlight python %}
def wait(current_lambda):
    return np.random.exponential(scale=1./current_lambda)
{% endhighlight %}

And, perhaps most importantly, a too to comute the size of the population alive at
any particular time, given the wgole history of our Cox process.

{% highlight python %}
def population_alive(history, current_time):
    result = 0
    n = len(history)
    for j in range(n):
        time_since_birth = current_time - history[j][0]
        lifespan = history[j][2]
        size = history[j][1]
        if time_since_birth < lifespan:
            result += size
    return  result
{% endhighlight %}

We initialize our history, intensity and time. We will also store all pairs [time, intensity]
taken at birth events.

{% highlight python %}
history = []
intensities =[]
intensities.append([0, lambda_0])
current_time = 0
intensity = lambda_0
{% endhighlight %}

The main loop goeas as follows.

{% highlight python %}
while current_time<time_limit:

    time_step = wait(intensity)
    if (current_time + time_step) > time_limit:
        break
    current_time += time_step
    newborn = make_birth(current_time + time_step)
    intensity = a(T) * (population_alive(history, current_time) + newborn[1]) + lambda_0  
    intensities.append([current_time, intensity])
    history.append(newborn)
    print('Time: ', current_time)
    print('Intensity: ', intensity)
    print()

intensities = np.transpose(intensities)
plt.plot(intensities[0], intensities[1])
plt.show()
{% endhighlight %}

We plot the intensities agains time. Notice, that we are mainly interested in
the scaling limit as  {% raw %}T{% endraw %} goes to infinity, so we may want
to set it to a very large number. Below we plot a sample path taken by the
intensity. Since the picture is automatically rescaled by matplotlib there is no point
in plotting the rescaled intensity.

![image tooltip here](/_assets/h1_image.PNG)
![your image]({{ site.url }}/_assets/h1_image.png)
















<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
