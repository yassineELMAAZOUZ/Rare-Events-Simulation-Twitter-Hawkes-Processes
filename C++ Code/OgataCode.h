#include <iostream>
#include <vector>
#include <fstream>


using namespace std;

void save_vector(const char* filename , vector<long double>& vect);
void save_vector(const char* filename , vector<long double>& vect){


    ofstream file_writer;

    file_writer.open(filename);



    for(int i =0; i<vect.size(); i++){

        file_writer<<vect[i]<<" ";

    }

    file_writer.close();
}




vector<long double> Ogata(long double* parameter);
vector<long double> Ogata(long double* parameter){


    long double T = parameter[0];
    long double lambda0 = parameter[1];
    long double alpha = parameter[2];
    long double beta = parameter[3];



    long double currentInstant = 0.0;            // the current generated instant.
    unsigned int numberOfInstants = 0;      // the current number of jumps.
    long double lambdaUpperBound = lambda0;      // the current upper bound for lambda_t for thining algorithm
    vector<long double> jumpTimes;               // a list of accepted instants (jumpTimes)

    long double jumpGapWidth = 0.0 , D = 0.0 , intensityOfNewPoint=0.0;


    while(currentInstant<T){

        long double u = (long double) rand()/RAND_MAX;
        jumpGapWidth = -log(u)/lambdaUpperBound;

        D = (long double) rand()/RAND_MAX;

        currentInstant += jumpGapWidth;

        intensityOfNewPoint = lambda0 + exp(-beta*jumpGapWidth)*(lambdaUpperBound-lambda0);


        if(D*lambdaUpperBound <= intensityOfNewPoint){

                lambdaUpperBound = intensityOfNewPoint + alpha;
                numberOfInstants+=1;
                jumpTimes.push_back(currentInstant);
        }
        else{

            lambdaUpperBound = intensityOfNewPoint;

        }
    }

    if(jumpTimes.back()>T){
        jumpTimes.pop_back();
    }


    return jumpTimes;

}


vector<long double> intensityCalculator(long double* parameter,vector<long double>& jumpTimes);
vector<long double> intensityCalculator(long double* parameter,vector<long double>& jumpTimes){

    long double T = parameter[0];
    long double lambda0 = parameter[1];
    long double alpha = parameter[2];
    long double beta = parameter[3];

    long double Mu = 0.0;
    vector<long double> intensities;
    intensities.push_back(Mu+lambda0);

    unsigned int N_T = jumpTimes.size();

    for(int i=0; i < N_T - 1; i++){

        Mu = exp( -beta * ( jumpTimes[i+1] - jumpTimes[i]) ) * ( Mu + alpha );
        intensities.push_back(lambda0 + Mu);

    }


    return intensities;

}

long double logLikelihood(long double* parameter,vector<long double>& jumpTimes);
long double logLikelihood(long double* parameter,vector<long double>& jumpTimes){

    long double T = parameter[0];
    long double lambda0 = parameter[1];
    long double alpha = parameter[2];
    long double beta = parameter[3];


    vector<long double> intensityValues = intensityCalculator(parameter,jumpTimes);

    long double firstTerm = 0.0;
    long double secondTerm = 0.0;

    for(int i=0; i<jumpTimes.size();i++){

        firstTerm = firstTerm + log(intensityValues[i]);

        secondTerm = secondTerm +  exp(  -beta*(T-jumpTimes[i])   ) - 1.;
    }


    return firstTerm + (alpha/beta)*secondTerm + T*(1-lambda0) ;



}

long double likely_Hood(long double* parameter1,long double* parameter2,vector<long double>& jumpTimes);
long double likely_Hood(long double* parameter1,long double* parameter2,vector<long double>& jumpTimes){


    long double logdP1_P0 = logLikelihood(parameter1,jumpTimes);
    long double logdP2_P0 = logLikelihood(parameter2,jumpTimes);

    return exp(logdP1_P0 - logdP2_P0);


}




long double asymptoticExpectedValue(long double* parameter);
long double asymptoticExpectedValue(long double* parameter){

    long double T = parameter[0];
    long double lambda0 = parameter[1];
    long double alpha = parameter[2];
    long double beta = parameter[3];


    if(alpha>=beta){
        return -1.;
    }

    return  (lambda0*T) / (1-(alpha/beta));



}



long double asymptoticStd(long double* parameter);
long double asymptoticStd(long double* parameter){

    long double T = parameter[0];
    long double lambda0 = parameter[1];
    long double alpha = parameter[2];
    long double beta = parameter[3];


    if(alpha>=beta){
        return -1.;
    }

    return sqrt( (lambda0*T) / pow( 1 - ( (double) alpha/beta) , 3 ) ) ;



}

long double getLambda0(long double T,long double alpha, long double beta,long double expectedValue);
long double getLambda0(long double T,long double alpha, long double beta,long double expectedValue){

    return (expectedValue/T)*(1-alpha/beta);

}






