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
  int numKeptSamples;
  int sizeObsVector;
  int currentIter;
  int accepted;
  double _shortTimeAccProbAvg;
  double _bLow, _bHigh;

  double *currentPhysicalState, *avgPhysicalState, *varPhysicalState, *_M2;
  double *proposedPhysicalState;
  double logLHDCurrentState;

  double accProb, avgAccProb;
  double rwmhStepSize, alphaPrior, priorVar, priorStd;

  // -- potentially not used --
  double *currentStateObservations, *proposedStateObservations;
  double *data;
  double currentLSQFunctional;
  double currentStateL2Norm2;
  double obsStdDev;  
  // --------------------------

  fftw_complex *currentSpectralState, *avgSpectralState;
  fftw_complex *priorDraw, *proposedSpectralState;

  fftw_plan _c2r;
  fftw_plan _r2c;

  gsl_rng *r;
};

typedef struct _mcmc_infchain mcmc_infchain;

void mcmc_init_infchain(mcmc_infchain *C, const int nj, const int nk);
void mcmc_free_infchain(mcmc_infchain *C);
void mcmc_reset_infchain(mcmc_infchain *C);
void mcmc_propose_RWMH(mcmc_infchain *C);
void mcmc_update_RWMH(mcmc_infchain *C, double logLHDOfProposal);
void mcmc_seed_with_prior(mcmc_infchain *C);
void mcmc_write_infchain_info(const mcmc_infchain *C, FILE *fp);
void mcmc_write_vectorfield_infchain(const mcmc_infchain *U, const mcmc_infchain *V, FILE *fp);
void mcmc_write_infchain(const mcmc_infchain *C, FILE *fp);
void mcmc_print_infchain(mcmc_infchain *C);
void mcmc_set_RWMH_stepsize(mcmc_infchain *C, double beta);
void mcmc_adapt_RWMH_stepsize(mcmc_infchain *C, double inc);
void mcmc_set_prior_alpha(mcmc_infchain *C, double alpha);
void mcmc_set_prior_var(mcmc_infchain *C, double var);
double mcmc_current_L2(mcmc_infchain *C);
double mcmc_proposed_L2(mcmc_infchain *C);
double mcmc_prior_L2(mcmc_infchain *C);

void mcmc_seed_with_divfree_prior(mcmc_infchain *C1, mcmc_infchain *C2);
void mcmc_propose_divfree_RWMH(mcmc_infchain *C1, mcmc_infchain *C2);
void mcmc_update_vectorfield_RWMH(mcmc_infchain *C1, mcmc_infchain *C2, double logLHDOfProposal);

void randomPriorDrawOLD(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs);

#endif
