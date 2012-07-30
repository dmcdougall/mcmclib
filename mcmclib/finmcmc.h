#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <Accelerate/Accelerate.h>

struct _FINCHAIN {
  int length; // Length of state vector
  int currentIter;
  
  double *currentState, *avgState, *varState, *proposedState, *_M2;
  double logLHDCurrentState;
  double accProb, avgAccProb;
  
  gsl_rng *r;
};

typedef struct _FINCHAIN FINCHAIN;

/*
  Initialises the finite dimensional chain
*/
void finmcmc_initChain(FINCHAIN *C, int n);

/*
  Frees the finite dimensional chain
*/
void finmcmc_freeChain(FINCHAIN *C);

void finmcmc_printChain(FINCHAIN C);

void finmcmc_writeChain(const FINCHAIN *C, FILE *fp);

void finmcmc_gausRanVec(const FINCHAIN *C, double *x, double stdDev);

/*
  Proposes a standard RWMH move
*/
void finmcmc_proposeRWMH(const FINCHAIN *C, double stdDev, double beta);

/*
  Update chain
*/
void finmcmc_updateRWMH(FINCHAIN *C, double logLHDOfProposal);

//void fin_initChain(FINCHAIN *C, int n, int sizePhi);

//void sampleRMWH(CHAIN *C);
//void sampleIndependenceSampler(CHAIN *C);
//void updateMean(CHAIN *C);
//void updateVar(CHAIN *C);
//void randomPriorDraw(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs);
