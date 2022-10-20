# [Shooting and Bouncing Rays](https://en.wikipedia.org/wiki/Shooting_and_bouncing_rays) solver 

Ray-tracing is based on [Havel-Herout ray-triangle intersection algorythm](https://www.researchgate.net/publication/41910471_Yet_Faster_Ray-Triangle_Intersection_Using_SSE4). The core of this solver is written in C using Numpy for memory allocation. Doing this allows to avoid creation of `n_triangles` by `n_rays` arrays.
