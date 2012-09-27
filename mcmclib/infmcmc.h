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
  int nj, nk; // number of Fourier coefficients in x/y direction respectively
  int num_kept_samples;
  int sizeObsVector;
  int current_iter;
  int accepted;
  double _short_time_acc_prob_avg;
  double _bLow, _bHigh;

  double *current_physical_state, *avg_physical_state, *var_physical_state, *_M2;
  double *proposed_physical_state;
  double log_likelihood_current_state;

  double acc_prob, avg_acc_prob;
  double rwmh_stepsize, alpha_prior, prior_var, prior_std;

  // -- potentially not used --
  double *current_state_observations, *proposed_state_observations;
  double *data;
  double current_LSQFunctional;
  double current_state_L2norm2;
  double obs_std_dev;
  // --------------------------

  fftw_complex *current_spectral_state, *avg_spectral_state;
  fftw_complex *prior_draw, *proposed_spectral_state;

  fftw_plan _c2r;
  fftw_plan _r2c;

  gsl_rng *r;
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
