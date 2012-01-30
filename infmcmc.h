#ifndef INFMCMC_H
#define INFMCMC_H

#include <complex.h>
#include <fftw3.h>

//void sampleRMWH(CHAIN *C);
//void sampleIndependenceSampler(CHAIN *C);
//void updateMean(CHAIN *C);
//void updateVar(CHAIN *C);

struct _CHAIN {
  int nj, nk; // number of Fourier coefficients in x/y direction respectively
  int numKeptSamples;
  int sizeObsVector;
  int currentIter;
  int accepted;
  
  double *currentPhysicalState, *avgPhysicalState, *varPhysicalState, *_M2;
  double *proposedPhysicalState;
  double logLHDCurrentState;
  double accProb, avgAccProb;
  double rwmhStepSize, alphaPrior;
  
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

typedef struct _CHAIN CHAIN;
typedef struct _CHAIN INFCHAIN;

void infmcmc_initChain(INFCHAIN *C, const int nj, const int nk);
void infmcmc_freeChain(INFCHAIN *C);
void infmcmc_proposeRWMH(INFCHAIN *C);
void infmcmc_updateRWMH(INFCHAIN *C, double logLHDOfProposal);
void infmcmc_seedWithPriorDraw(INFCHAIN *C);
void infmcmc_writeChainInfo(const INFCHAIN *C, FILE *fp);
void infmcmc_writeVFChain(const INFCHAIN *U, const INFCHAIN *V, FILE *fp);
void infmcmc_writeChain(const INFCHAIN *C, FILE *fp);
void infmcmc_printChain(INFCHAIN *C);
void infmcmc_setRWMHStepSize(INFCHAIN *C, double beta);
void infmcmc_setPriorAlpha(INFCHAIN *C, double alpha);
double infmcmc_L2Current(INFCHAIN *C);
double infmcmc_L2Proposed(INFCHAIN *C);
double infmcmc_L2Prior(INFCHAIN *C);

void infmcmc_seedWithDivFreePriorDraw(INFCHAIN *C1, INFCHAIN *C2);
void infmcmc_proposeDivFreeRWMH(INFCHAIN *C1, INFCHAIN *C2);
void infmcmc_updateVectorFieldRWMH(INFCHAIN *C1, INFCHAIN *C2, double logLHDOfProposal);

void randomPriorDrawOLD(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs);

#endif
