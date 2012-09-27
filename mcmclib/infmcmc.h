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

void infmcmc_initChain(mcmc_infchain *C, const int nj, const int nk);
void infmcmc_freeChain(mcmc_infchain *C);
void infmcmc_resetChain(mcmc_infchain *C);
void infmcmc_proposeRWMH(mcmc_infchain *C);
void infmcmc_updateRWMH(mcmc_infchain *C, double logLHDOfProposal);
void infmcmc_seedWithPriorDraw(mcmc_infchain *C);
void infmcmc_writeChainInfo(const mcmc_infchain *C, FILE *fp);
void infmcmc_writeVFChain(const mcmc_infchain *U, const mcmc_infchain *V, FILE *fp);
void infmcmc_writeChain(const mcmc_infchain *C, FILE *fp);
void infmcmc_printChain(mcmc_infchain *C);
void infmcmc_setRWMHStepSize(mcmc_infchain *C, double beta);
void infmcmc_adaptRWMHStepSize(mcmc_infchain *C, double inc);
void infmcmc_setPriorAlpha(mcmc_infchain *C, double alpha);
void infmcmc_setPriorVar(mcmc_infchain *C, double var);
double infmcmc_L2Current(mcmc_infchain *C);
double infmcmc_L2Proposed(mcmc_infchain *C);
double infmcmc_L2Prior(mcmc_infchain *C);

void infmcmc_seedWithDivFreePriorDraw(mcmc_infchain *C1, mcmc_infchain *C2);
void infmcmc_proposeDivFreeRWMH(mcmc_infchain *C1, mcmc_infchain *C2);
void infmcmc_updateVectorFieldRWMH(mcmc_infchain *C1, mcmc_infchain *C2, double logLHDOfProposal);

void randomPriorDrawOLD(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs);

#endif
