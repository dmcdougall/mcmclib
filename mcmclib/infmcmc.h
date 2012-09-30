#ifndef INFMCMC_H
#define INFMCMC_H

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>

//void sampleRMWH(CHAIN *C);
//void sampleIndependenceSampler(CHAIN *C);
//void updateMean(CHAIN *C);
//void updateVar(CHAIN *C);

struct _mcmc_infchain {
  /**
   * Number of degrees of freedom (Fourier coefficients) in the x direction.
   */
  int nj;
  /**
   * Number of degrees of freedom (Fourier coefficients) in the y direction.
   */
  int nk;
  /**
   * Stores the current Markov chain iteration number.
   */
  int current_iter;
  /**
   * Number of samples you wish to keep in the Markov chain.
   */
  int num_kept_samples;
  /**
   * This is 1 if the previous iteration was accepted and 0 if it was rejected.
   */
  int accepted;
  /**
   * Current Markov chain state in the physical domain.
   */
  double *current_physical_state;
  /**
   * Average Markov chain state in the physical domain.
   */
  double *avg_physical_state;
  /**
   * Variance of the Markov chain state in the physical domain.
   */
  double *var_physical_state;
  /**
   * Stores the most recently proposed Markov chain state in the physical
   * domain.
   */
  double *proposed_physical_state;
  /**
   * Current Markov chain state in the coefficient domain.
   */
  fftw_complex *current_spectral_state;
  /**
   * Average Markov chain state in the coefficient domain.
   */
  fftw_complex *avg_spectral_state;
  /**
   * Stores the most recently proposed Markov chain state in the coefficient
   * domain.
   */
  fftw_complex *proposed_spectral_state;
  /**
   * Stores a draw from the prior distribution in coefficient domain.
   */
  fftw_complex *prior_draw;
  /**
   * The current acceptance probability.
   */
  double acc_prob;
  /**
   * The empirical mean acceptance probability. This is the acceptance rate.
   */
  double avg_acc_prob;
  /**
   * The random-walk proposal step size.
   */
  double rwmh_stepsize;
  /**
   * The fractional power of the inverse of the Laplacian operator. This
   * operator is the covariance operator of the prior distribution.
   */
  double alpha_prior;
  /**
   * Multiplicative coefficient of the prior covariance operator. Larger
   * values mean larger variance. Smaller values mean smaller variance.
   */
  double prior_var;
  /**
   * Square root of the multiplicative coefficient of the prior variance
   * operator.
   */
  double prior_std;
  /**
   * Stores the log likelihood of the current state. You should set this after
   * proposing a state.
   */
  double log_likelihood_current_state;
  /**
   * The size of the output from the energy functional. The energy functional
   * may be referred to as a different name depending on your scientific field.
   */
  int sizeObsVector;
  /**
   * Stores the output of the energy functional using the current state as
   * input.
   */
  double *current_state_observations;
  /**
   * Stores the output of the energy functional using the proposed state as
   * input.
   */
  double *proposed_state_observations;
  /**
   * CHECK THIS ISN'T NEEDED
   */
  double *data;
  /**
   * Stores the current value of the least-squares functional. This is
   * potentially not needed.
   */
  double current_LSQFunctional;
  /**
   * The standard deviation of errors in the observations. This is potentially
   * not needed.
   */
  double obs_std_dev;
  /**
   * Stores the square of the L2 norm of the current Markov chain state.
   */
  double current_state_L2norm2;
  /**
   * Private
   */
  fftw_plan _c2r;
  /**
   * Private
   */
  fftw_plan _r2c;
  /**
   * Private
   */
  gsl_rng *r;
  /**
   * Private
   */
  double _short_time_acc_prob_avg;
  /**
   * Private
   */
  double _bLow;
  /**
   * Private
   */
  double _bHigh;
  /**
   * Private
   */
  double *_M2;
};

typedef struct _mcmc_infchain mcmc_infchain;

void mcmc_init_infchain(mcmc_infchain *chain, const int nj, const int nk);
void mcmc_free_infchain(mcmc_infchain *chain);
void mcmc_reset_infchain(mcmc_infchain *chain);
void mcmc_propose_RWMH(mcmc_infchain *chain);
void mcmc_update_RWMH(mcmc_infchain *chain, double logLHDOfProposal);
void mcmc_seed_with_prior(mcmc_infchain *chain);
void mcmc_write_infchain_info(const mcmc_infchain *chain, FILE *fp);
void mcmc_write_vectorfield_infchain(const mcmc_infchain *U, const mcmc_infchain *V, FILE *fp);
void mcmc_write_infchain(const mcmc_infchain *chain, FILE *fp);
void mcmc_print_infchain(mcmc_infchain *chain);
void mcmc_set_RWMH_stepsize(mcmc_infchain *chain, double beta);
void mcmc_adapt_RWMH_stepsize(mcmc_infchain *chain, double inc);
void mcmc_set_prior_alpha(mcmc_infchain *chain, double alpha);
void mcmc_set_prior_var(mcmc_infchain *chain, double var);
double mcmc_current_L2(mcmc_infchain *chain);
double mcmc_proposed_L2(mcmc_infchain *chain);
double mcmc_prior_L2(mcmc_infchain *chain);

void mcmc_seed_with_divfree_prior(mcmc_infchain *chain1, mcmc_infchain *chain2);
void mcmc_propose_divfree_RWMH(mcmc_infchain *chain1, mcmc_infchain *chain2);
void mcmc_update_vectorfield_RWMH(mcmc_infchain *chain1, mcmc_infchain *chain2, double logLHDOfProposal);

void randomPriorDrawOLD(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs);

#endif
