Smooth local projections
========================

Barnichon and Brownlees [BB2019]_ propose smooth local projections (SLP) as
a way to estimate impulse responses more precisely than standard local
projections. A standard local projection estimates a separate coefficient at
every horizon. This is flexible, but the resulting impulse response can be
noisy because neighboring horizons are estimated independently. SLP retains
the local-projection regression while allowing information to be shared across
horizons through a penalized B-spline representation.


Reference
---------

Here I first describe in very broad terms what the paper does.
The next section describes my implementation in detail.

Local projections
^^^^^^^^^^^^^^^^^

Let :math:`y_t` be an outcome, :math:`x_t` the shock or treatment of interest,
and :math:`w_{it}` a collection of controls. The local projection at horizon
:math:`h` is

.. math::
   :label: slp-local-projection

   y_{t+h}
   = \alpha(h) + \beta(h)x_t
   + \sum_{i=1}^{p}\gamma_i(h)w_{it}
   + u_{t+h}^{(h)}.

The sequence :math:`\beta(h)` is the dynamic multiplier, or impulse response,
of :math:`y` to :math:`x`. Estimating :eq:`slp-local-projection` separately for
every horizon places no smoothness restriction on this sequence, but can
produce estimates with substantial sampling variation.

B-spline approximation
^^^^^^^^^^^^^^^^^^^^^^

SLP approximates each horizon-dependent coefficient with a linear combination
of :math:`K` B-spline basis functions. For the dynamic multiplier,

.. math::
   :label: slp-basis-expansion

   \beta(h) \approx \sum_{k=1}^{K} b_k B_k(h)
   = B(h)^{\mathsf T}b.

The intercept and control coefficients are expanded in the same way. The basis
functions are local, overlapping, piecewise polynomials. This gives a flexible
representation of the impulse response while keeping the model linear in the
unknown spline coefficients.

.. figure:: _static/slp-b-spline-basis-000.png
   :alt: A set of overlapping cubic B-spline basis functions.
   :align: center
   :width: 95%

   Cubic B-spline basis used by Barnichon and Brownlees. The emphasized curve
   is the basis function associated with knot 6. Source: figure 1 of
   [BB2019]_.

After substituting the basis expansions into
:eq:`slp-local-projection`, the regressions for all horizons can be stacked as

.. math::
   :label: slp-stacked-model

   Y_t = X_t\theta + U_t,

where :math:`\theta` collects the spline coefficients for the intercept, shock,
and controls. Vertically stacking observations over :math:`t` gives
:math:`Y=X\theta+U`.

Penalized estimation
^^^^^^^^^^^^^^^^^^^^

The spline coefficients are estimated by generalized ridge regression:

.. math::
   :label: slp-penalized-estimator

   \widehat{\theta}
   &= \operatorname*{arg\,min}_{\theta}
      \left\{
          \lVert Y-X\theta\rVert_2^2
          + \lambda\theta^{\mathsf T}P\theta
      \right\} \\
   &= \left(X^{\mathsf T}X+\lambda P\right)^{-1}X^{\mathsf T}Y.

The penalty acts on finite differences between adjacent spline coefficients.
If :math:`D_r` represents the :math:`r`-th difference operator, the penalty for
the impulse-response coefficients is

.. math::
   :label: slp-difference-penalty

   \lambda\sum_{k=r+1}^{K}\left(\Delta^r b_k\right)^2
   = \lambda (D_r b)^{\mathsf T}(D_r b)
   = \lambda b^{\mathsf T}D_r^{\mathsf T}D_r b.

Thus, :math:`P=D_r^{\mathsf T}D_r` for a single coefficient curve, with a
block-diagonal extension when several curves are penalized. A small
:math:`\lambda` leaves the horizon coefficients relatively unrestricted and
approaches ordinary local projections. A large :math:`\lambda` drives the
:math:`r`-th differences toward zero and therefore shrinks the response toward
a polynomial of order :math:`r-1`. The paper selects :math:`\lambda` by
:math:`k`-fold cross-validation.

Newey--West covariance estimate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because stacked, overlapping forecast errors are serially correlated, the
paper uses a Newey--West covariance estimate. Define

.. math::

   A_{\lambda}
   = \sum_t X_t^{\mathsf T}X_t + \lambda P,
   \qquad
   s_t = X_t^{\mathsf T}\widehat U_t,

and the lag-:math:`\ell` score covariance

.. math::

   \widehat\Gamma_{\ell}
   = \frac{1}{T}\sum_{t=\ell+1}^{T-H_{\min}}
     s_t s_{t-\ell}^{\mathsf T}.

Here, :math:`T` is the input series length and :math:`H_{\min}` is the smallest
projection horizon, typically either 0 (in our setting) or 1.
With Bartlett weights :math:`w_{\ell}=1-\ell/(L+1)`, the estimator is

.. math::
   :label: slp-newey-west

   \widehat V(\widehat\theta)
   = T A_{\lambda}^{-1}
     \left[
       \widehat\Gamma_0
       + \sum_{\ell=1}^{L}w_{\ell}
         \left(\widehat\Gamma_{\ell}
         + \widehat\Gamma_{\ell}^{\mathsf T}\right)
     \right]
     A_{\lambda}^{-1}.

Barnichon and Brownlees set the bandwidth to :math:`L=H`, the maximum projection horizon. For inference they
use a penalty equal to one tenth of the degree of shrinkage selected by
cross-validation. If :math:`\widehat b` and
:math:`\widehat V(\widehat b)` denote the impulse-response spline coefficients
and their covariance block, the point estimate and its pointwise standard
error at horizon :math:`h` are

.. math::

   \widehat\beta(h) = B(h)^{\mathsf T}\widehat b,
   \qquad
   \operatorname{se}\!\left(\widehat\beta(h)\right)
   = \sqrt{B(h)^{\mathsf T}
           \widehat V(\widehat b)B(h)}.

The authors describe this inference procedure as heuristic, particularly
because selecting the smoothing parameter from the data complicates the
distribution of the shrinkage estimator.



Implementation
--------------


Masking Observations
^^^^^^^^^^^^^^^^^^^^

The primary change that we need to implement is the masking of observations.
An intuitive implementation is to simply cut these observations out before
giving the data to the estimation method.
However, since the package assumes one continuous time series, this will effectively
treat observations from right before and after the break as being closely related.

A better approach is to mask observations that fall within the window directly
within the estimation package.
When a mask is provided, I omit an observation if :math:`t`, :math:`t+h`
or any observation in between is masked.
In the following (and in the code), :math:`m_{th}` indicates whether an observation
can be used, so its meaning is the opposite of the mask in the public function interface.


Spline Support
^^^^^^^^^^^^^^

One difference between the paper and my code is that in the paper the splines start at -2,
while I use one additional spline.
This corresponds to the code that the authors provide and ensures that the splines sum to 1 at each
interior knot.
This small change improves how the penalty is applied.


Partialling Out Controls
^^^^^^^^^^^^^^^^^^^^^^^^

As the paper mentions, we can apply :eq:`slp-basis-expansion` to all parameters or only a subset of them.
I implement this through partialling out controls.
If selected, variables passed as controls are not expanded in the spline basis, but instead partialled out before
the penalized fit.
Since the expansion and smoothing penalty search are quite expensive, this provides a substantial performance boost,
in particular for a large number of controls.
My understanding is that when no penalty is applied, these two estimation proceedures produce identical results up to numeric precision
(in theory, in practice without applying a penalty the estimator is no longer well defined).

Furthermore, my code selects only a single penalty parameter. Restricting the penalty to apply only to the
IRF of interest therefore means that the penalty is selected optimally for this IRF, not for all IRFs.
This might be particularly important if we have a large number of controls, many of which have little to no
additional impact and thus a relatively high ratio of noise to signal.


Defining the Penalty and Gram Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Especially when not partialling out controls and at longer horizons, these two matrices become quite large.
For example, at :math:`H=100` with 90 controls, we have about 9000 expanded coefficients.
With 2000 observations, the full stacked control matrix defined in the paper would consume over 13 GB of memory.

I can, however, avoid materializing this matrix and also greatly reduce the storage required for the
:math:`9000\times9000` Gram matrix.
The key is that, in the expanded form, only closely adjacent spline basis functions overlap.
For example, splines 3 and 15 have disjoint support, so at most one can be nonzero at each horizon :math:`h`.

Start by defining the expanded regressors as :math:`\tilde{x}_{th} = b_h \otimes x_t`,
where :math:`b_h` evaluates the :math:`K` splines at horizon :math:`h`.
From this :math:`\tilde{x}_{th} \tilde{x}_{th}^T = (b_h b_h^T) \otimes (x_t x_t^T)`.
The key is that :math:`b_h b_h^T` is not a dense matrix, but has non-zero entries only on the main diagonal
and the first two off-diagonals on either side at the integer horizons used here
(when fitting cubic splines, which is the only setting I currently support).

From this I can compute the Gram matrix :math:`X^TX = \sum_{h=0}^{H} \sum_{t=1}^T m_{th} \tilde x_{th} \tilde x_{th}^T`.
The entire matrix is thus a block-banded matrix and since it is symmetric I only ever compute the entries on and below the diagonal.
This greatly reduces the computation and storage required for this matrix.

Similarly, I define the penalty matrix as :math:`P = (D^{\mathsf T} D) \otimes I` which gives a similar block-banded structure.
The matrix :math:`D` is computed through repeated differencing of the identity matrix,
just like in the R-script provided by the authors.

Together with the product :math:`X^{\mathsf T} y` these are the main inputs for the penalty fitting step.


Penalty Fitting
^^^^^^^^^^^^^^^

I look for the optimal penalty across a grid that by default ranges from :math:`10^{-3}` to :math:`10^3`.
The base-10 logarithms of the grid points are evenly spaced, and by default I evaluate 1000 grid points. This is a rather wide grid,
but I have little guidance on the typical range and often found even quite extreme values.
I scale the penalty internally by :math:`N`, which is the total number of observations
(not the length of the input series, but the actual number of observations in the smoothing step,
typically close to the input length times the maximum horizon).
This is intended to make the grid adjust for the sample size.

For tuning I mostly use two measures, a generalized cross-validation error and a 5-fold cross-validation error.
The first is much faster, while the second is closer to the algorithm suggested by the authors.

Both approaches effectively compute the RSS, so in order to make evaluating so many grid points computationally
feasible, I need a cheap way of finding the fitted values for any given smoothing parameter.
For this I use the Demmler–Reinsch orthogonalization.
I first describe this method in general, before going into implementation details of the two tuning approaches.

My Demmler–Reinsch implementation closely follows Ruppert, Wand and Carroll [RWC2003]_.
One notable difference is that my Gram matrix is singular, since there are more splines than horizons.
To address this, I add a multiple of the penalty matrix to the Gram matrix.
Below I show the algorithm with this modification.

The goal is to compute fitted values

.. math::

   \hat y_\lambda = X (X^{\mathsf T} X + \lambda P)^{-1} X^{\mathsf T} y

for many different values of :math:`\lambda`.
In the formulas below, :math:`\lambda` denotes the internally scaled penalty.
The method does so by rotating the design such that applying the penalty is a simple multiplication.

First, define :math:`L` as the Cholesky factor of

.. math::

   X^{\mathsf T}X + \overline{\lambda} P = L L^{\mathsf T}.

We can then rewrite

.. math::

   X^{\mathsf T}X + \lambda P
   &= L I_n L^{\mathsf T}
      + (\lambda - \overline{\lambda}) LL^{-1} P L^{-\mathsf T}L^{\mathsf T} \\
   &= L U I_n U^{\mathsf T} L^{\mathsf T}
      + (\lambda - \overline{\lambda}) L U S U^{\mathsf T} L^{\mathsf T}\\
   &= L U ( I + (\lambda - \overline \lambda) S )U^{\mathsf T} L^{\mathsf T}


where the last statement uses the eigenvalue decomposition of

.. math::

   L^{-1} P L^{-\mathsf T} = U S U^{\mathsf T}

and :math:`S` is a diagonal matrix containing the eigenvalues and :math:`U` is orthogonal.
This gives the fitted values as

.. math::

   \hat y_\lambda
   &= X L^{-\mathsf T} U (I_n + (\lambda-\overline{\lambda}) S)^{-1}
      U^{\mathsf T} L^{-1} X^{\mathsf T} y \\
   &= A (I_n + (\lambda-\overline{\lambda}) S)^{-1} A^{\mathsf T} y,

with :math:`A = X L ^{-\mathsf T} U`.
Here, :math:`I_n` is the identity matrix with dimension equal to the number of spline coefficients.
Let :math:`b=A^{\mathsf T}y` and let :math:`s_j` be the :math:`j`-th diagonal entry of :math:`S`.
Then :math:`d=(I_n+(\lambda-\overline\lambda)S)^{-1}b` has entries
:math:`d_j=b_j/[1+(\lambda-\overline\lambda)s_j]`.

This greatly reduces the cost of computing the residual sum of squares (RSS):

.. math::

   RSS
   &= (y-\hat y)^{\mathsf T} (y-\hat y) \\
   &= y^{\mathsf T}y
      - 2 y^{\mathsf T} A d
      + d^{\mathsf T} A^{\mathsf T}A d \\
   &= y^{\mathsf T}y
      - 2 b^{\mathsf T} d
      + d^{\mathsf T} \left( I_n-\overline\lambda S \right) d

where the last line uses the fact that

.. math::
    A^{\mathsf T} A &= U^{\mathsf T}L^{-1} X^{\mathsf T} X L^{-\mathsf T} U\\
    & = U^{\mathsf T}L^{-1} \left( L L^{\mathsf T} - \overline\lambda P \right) L^{-\mathsf T} U\\
    & = I_n - \overline \lambda U^{\mathsf T} U S U^{\mathsf T} U = I_n - \overline\lambda S

With this, evaluating a single grid point :math:`\lambda` requires only cheap multiplications and vector products.

The GCV criterion can then be computed from

.. math::

    GCV(\lambda) = RSS(\lambda) / (1 - d_{fit}(\lambda) / N)^2

where :math:`d_{fit}(\lambda) = \sum_{j=1}^{n} \frac{1-\overline\lambda s_j}{1+(\lambda-\overline\lambda)s_j}`
is the effective number of fitted parameters in the penalized fit. When controls are
partialled out, their fitted degrees of freedom are added to this quantity.


For the 5-fold cross-validation tuning, I first compute the stacked :math:`X` and :math:`Y` arrays.
I then count the number of available observations along each horizon and then split each horizon into 5
equally sized buckets.
However, a bucket for the first horizon is larger than one for the last horizon.
So an observation might fall in bucket 1 at horizon 0 and bucket 2 at horizon 100.

With this grouping, for each bucket I use the observations of all other buckets as the training sample
and the held-out bucket for evaluation.
I perform a Demmler–Reinsch orthogonalization like for the GCV tuning,
but with the slight difference that fitted values are only computed for the validation data.
Using tildes to denote validation data, I compute

.. math::

    \hat{\tilde y} = \tilde X (X^{\mathsf T} X + \lambda P)^{-1} X^{\mathsf T} y

The further steps of the algorithm then follow immediately, with the single exception that
:math:`\tilde A^{\mathsf T}\tilde A` is no longer a diagonal matrix,
where :math:`\tilde A=\tilde X L^{-\mathsf T}U`.

In general, I've heavily optimized this code.
The Cholesky decompositions can exploit the sparsity of the Gram and penalty matrices and are thus rather fast.
But since the inverse of the Cholesky factor is a dense matrix
(at least I'm not aware of any sparse structure in this matrix),
the eigenvalue decomposition is performed on a dense matrix and thus likely the most expensive step of this
algorithm.



Covariance Estimation
^^^^^^^^^^^^^^^^^^^^^

Here I'm not sure if I understand the paper correclty.
They say that they use a penalty scaled by 0.1 in the covariance estimation.
But they also state in a footnote, that smoothing estimators typically have some bias and that constructing CIs with an undersmoothed version can reduce this bias.
This would mean refitting the parameter vector and ultimately lead to CIs that are note centered on the actual estiamte.
They however never state that they do suggest this.
I don't go so far, I just undersmooth in the sandwich estimator, but do not reestimate the parameter vector.
Here their R and matlab code also seem to disagree, with matlab reestimating and R not even undersmoothen the sandwich.

Note that when partialling out controls, I estimate only the covariance matrix for the shocks.
Similarly, when keeping controls in the main estimation, I'm not interested in estimating covariances of these controls.
To avoid estimating these, I define a matrix :math:`R` that selects only the relevant entries of the full parameter
vector.
I'm thus interested in :math:`\tilde \theta = R \theta` with covariance matrix :math:`V(\tilde\theta) = R V(\theta) R^{\mathsf T}`.
This allows me to rewrite the covariance matrix to compute the long-run variance only on a subset of the full design vector:

.. math::

    W &= R(X^{\mathsf T}X + 0.1\lambda P)^{-1},
    \qquad \tilde X_t = X_t W^{\mathsf T}, \\
    C_l &= \sum_t X_t^{\mathsf T}\widehat U_t
                  \widehat U_{t-l}^{\mathsf T}X_{t-l}.

Here, :math:`C_l` is an unnormalized lag covariance sum (equal to
:math:`T\widehat\Gamma_l` above), and each sum includes only available time pairs.
Then

.. math::

    \widehat V(\tilde\theta)
    &= W \left[ C_0 + \sum_{l=1}^{H} w_l
                \left(C_l+C_l^{\mathsf T}\right) \right] W^{\mathsf T}\\
    &= \sum_t \tilde X_t^{\mathsf T}\widehat U_t
                     \widehat U_t^{\mathsf T}\tilde X_t \\
    &\quad + \sum_{l=1}^{H} w_l \sum_t
       \left[\tilde X_t^{\mathsf T}\widehat U_t
             \widehat U_{t-l}^{\mathsf T}\tilde X_{t-l}
       + \tilde X_{t-l}^{\mathsf T}\widehat U_{t-l}
             \widehat U_t^{\mathsf T}\tilde X_t\right].

Since :math:`X_t` has dimensions :math:`(H+1)\times k`, where :math:`k` is the number of shocks and controls multiplied by the number of splines,
:math:`\tilde X_t` may be much smaller since its second dimension is only the number of shocks times the number of splines.



References
----------

.. [BB2019] Regis Barnichon and Christian Brownlees. “Impulse Response
   Estimation by Smooth Local Projections.” *The Review of Economics and
   Statistics*, 101(3), 2019, pp. 522–530.
   https://doi.org/10.1162/rest_a_00778

.. [RWC2003] David Ruppert, M. P. Wand, and R. J. Carroll.
   *Semiparametric Regression.* Cambridge University Press, 2003.
