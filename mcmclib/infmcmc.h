/**
 * \file infmcmc.h
 * \brief Public API to infinite dimensional Markov chain
 *
 * Functions are provided to interface with the Markov chain data
 * structure.
 */
#ifndef INFMCMC_H
#define INFMCMC_H

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>

#include "prior_general.h"

#define MCMC_INFCHAIN_GENERAL (0U)
#define MCMC_INFCHAIN_PERIODIC (1U << 0)

/**
 * \struct _mcmc_infchain
 * \brief The Markov chain data structure
 */
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
   * Number of (general) degrees of freedom
   */
  int ndofs;
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
  fftw_plan _c2r;
  fftw_plan _r2c;
  gsl_rng *r;
  double _short_time_acc_prob_avg;
  double _bLow;
  double _bHigh;
  double *_M2;
  unsigned int _type;
  prior_data *_prior;
  void (* _to_physical)(struct _mcmc_infchain *c, void *source, void *dest);
  void (* _to_coefficient)(struct _mcmc_infchain *c, void *source, void *dest);
};

typedef struct _mcmc_infchain mcmc_infchain;

/**
 * \brief Initialise an infinite dimensional Markov chain.
 *
 * The Markov Chain is assumed to live on a unifromly gridded square with
 * periodic boundary conditions. These assumptions allow fast tranforms between
 * the physical- and coefficient-domains.
 *
 * This function needs to be called before any other mcmclib function. It
 * allocates memory needed by some of the outher functions.
 *
 * \param chain
 * The chain to initialise.
 *
 * \param nj
 * Number of Fourier coefficients in the x direction.
 *
 * \param nk
 * Number of Fourier corfficients in the y direction.
 */
void mcmc_init_infchain(mcmc_infchain *chain, unsigned int type, const int nj, const int nk);

/**
 * \brief Free a previously initialised Markov chain
 *
 * This frees all the memory allocated by ::mcmc_init_infchain.
 *
 * \param chain
 * The chain to free.
 */
void mcmc_free_infchain(mcmc_infchain *chain);

/**
 * \brief Re-initialise all heap-allocated memory in the given chain.
 *
 * This is equivalent to a call to ::mcmc_free_infchain followed by a call to
 * ::mcmc_init_infchain.
 *
 * \param chain
 * The chain to reset.
 */
void mcmc_reset_infchain(mcmc_infchain *chain);

/**
 * \brief Make a standard random-walk proposal.
 *
 * \param chain
 * The chain to make the proposal.
 */
void mcmc_propose_RWMH(mcmc_infchain *chain);

/**
 * \brief Update the chain after a proposal has been made.
 *
 * Given a chain that has made a proposal using ::mcmc_propose_RWMH and the
 * value of the log-likelihood at the proposed state, update the current state
 * of the chain and its moments.
 *
 * \param chain
 * The chain to update.
 *
 * \param logLHDOfProposal
 * The value of the log-likelihood functional using
 * _mcmc_infchain::proposed_physical_state as input.
 */
void mcmc_update_RWMH(mcmc_infchain *chain, double logLHDOfProposal);

/**
 * \brief Seed the current state of the Markov chain with a random draw from
 * the prior distribution.
 *
 * \param chain
 * The chain to seed
 */
void mcmc_seed_with_prior(mcmc_infchain *chain);

/**
 * \brief Write chain information to disk.
 *
 * This only writes the _mcmc_infchain::nj and _mcmc_infchain::nk
 * parameters to disk.
 *
 * \param chain
 * The chain whose information is to be written to disk.
 *
 * \param fp
 * The file pointer to write to.
 */
void mcmc_write_infchain_info(const mcmc_infchain *chain, FILE *fp);

/**
 * \brief Write out two infinite dimensional Markov chains (a vector field)
 *
 * \param U
 * The horizontal component of the vector field
 *
 * \param V
 * The vertical component of the vector field
 *
 * \param fp
 * The file pointer to write the data to
 */
void mcmc_write_vectorfield_infchain(const mcmc_infchain *U, const mcmc_infchain *V, FILE *fp);

/**
 * \brief Write out Markov chain data to disk
 *
 * \param chain
 * The chain to write to disk
 *
 * \param fp
 * The file pointer to write the data to
 */
void mcmc_write_infchain(const mcmc_infchain *chain, FILE *fp);

/**
 * \brief Print some diagnostic chain information to the screen.
 *
 * \param chain
 * The chain whose information is printed to the screen.
 */
void mcmc_print_infchain(mcmc_infchain *chain);

/**
 * \brief Set the random-walk proposal step size of a Markov chain
 *
 * \param chain
 * The chain whose random-walk proposal step size is altered
 *
 * \param beta
 * The step size to change to
 */
void mcmc_set_RWMH_stepsize(mcmc_infchain *chain, double beta);

/**
 * \brief Adapt the random-walk proposal of a Markov chain
 *
 * Adapt the Markov chain's random-walk proposal step size. It is adapted by
 * either +inc or -inc depening on whether a current short-time average admits
 * an acceptance rate that is too high or too low, respectively.
 *
 * 'Too high' is currently hard-coded at more than 30%. 'Too low' is hard-coded
 * at less than 20%.
 *
 * Note: altering a Markov chain's proposal distribution destroys the Markovian
 * property of the chain. Ergodicity is no longer guaranteed under these
 * circumstances. Use this function with care.
 *
 * \param chain
 * The chain to adapt
 *
 * \param inc
 * The increment to apply to the chain
 */
void mcmc_adapt_RWMH_stepsize(mcmc_infchain *chain, double inc);

/**
 * \brief Set the fractional power of the prior covariance operator.
 *
 * The prior covariance operator is assumed to be the inverse of the Laplacian
 * operator.
 *
 * \param chain
 * The chain to apply the change to
 *
 * \param alpha
 * The power to which the inverse Laplacian is raised
 */
void mcmc_set_prior_alpha(mcmc_infchain *chain, double alpha);

/**
 * \brief Set the Markov chain's prior covariance coefficient
 *
 * A larger prior covariance implies a less informative prior distribution.
 * A smaller prior covariance implies a more informative prior.
 *
 * \param chain
 * The chain to apply the change to
 *
 * \param var
 * The multiplicative coefficient of the prior covariance operator
 */
void mcmc_set_prior_var(mcmc_infchain *chain, double var);

/**
 * \brief Compute the square of the L2 norm of the current chain state
 *
 * \param chain
 * The chain whose current state is used to compute the norm
 *
 * \return
 * The L2 norm of the current state of the chain
 */
double mcmc_current_L2(mcmc_infchain *chain);

/**
 * \brief Compute the square of the L2 norm of the proposed chain state
 *
 * \param chain
 * The chain whose proposed state is used to compute the norm
 *
 * \return
 * The L2 norm of the proposed state of the chain
 */
double mcmc_proposed_L2(mcmc_infchain *chain);

/**
 * \brief Compute the square of the L2 norm of the currently stored prior draw
 *
 * \param chain
 * The chain whose prior draw state is used to compute the norm
 *
 * \return
 * The L2 norm of the currently stored prior draw
 */
double mcmc_prior_L2(mcmc_infchain *chain);

/**
 * \brief Seed two Markov chains with divergence free vector field
 *
 * After calling this function, the current state of chain1 and the current
 * state of chain 2 form a vector field. This vector field will be divergence
 * free.
 *
 * \param chain1
 * The chain that will hold the horizontal component of the vector field
 *
 * \param chain2
 * The chain that will hold the vertical component of the vector field
 */
void mcmc_seed_with_divfree_prior(mcmc_infchain *chain1, mcmc_infchain *chain2);

/**
 * \brief Propose a divergence free vector field
 *
 * After calling this function, the proposed state of chain1 and chain2 will
 * describe a vector field whose divergence is zero.
 *
 * \param chain1
 * The chain to hold the horizontal component
 *
 * \param chain2
 * The chain to hold the vertical component
 */
void mcmc_propose_divfree_RWMH(mcmc_infchain *chain1, mcmc_infchain *chain2);

/**
 * \brief Update a divergence free vector field using random-walk
 * Metropolis-Hastings
 *
 * Given the value of the log-likelihood at proposed vector field, update the
 * current state of both chains
 *
 * TODO: Add information about what members are updated
 *
 * \param chain1
 * The horizontal component of the vector field
 *
 * \param chain2
 * The vertical component of the vector field
 *
 * \param logLHDOfProposal
 * The value of the log-likelihood at the proposed vector field
 */
void mcmc_update_vectorfield_RWMH(mcmc_infchain *chain1, mcmc_infchain *chain2, double logLHDOfProposal);

// Prior stuff
int mcmc_infchain_set_prior_data(mcmc_infchain *chain, double *evals,
    double *evecs);

void randomPriorDrawOLD(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs);
#endif
