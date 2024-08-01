
# 1D and 2D Filtering

I would first devloped an array using numpy to create a 1D filter or a 2d filter, then I will apply the the constant of the
gaussian function which is 1/2*pi*sigma^2 throughout the entire filter that I have created. Then I would create three if statement
that would determine whether the filter is a 1d or 2d filter by check its dimension.Afterwards I would use a for loop that iterates through
the filter and compute the last portion of the gaussian filter where I would apply the equation and then I will apply the rest of the gaussian
to each pixel of an image creating me a gaussian function

# Edge Detection

Applying  the edge detection incorperate a lot of my previous filters that I have created throughout this project such as applying
a gaussian filter which smoothed the image, reducing noise and allowing me to devlop a clean edge detection. I decided to use filter
which is a dervivative filter and I applied both towards the x and y axis. By doing so I computed the graident of both axis and created a simple
threshold through process of elimination to find the best results. It didn't come out as great as I had expected and I believe one of the reason is
I didn't calculate the direction of the gradient and instead I only used the magnitude which led to my results.
