#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>
#include <boost/math/distributions/normal.hpp>
#include "OgataCode.h"

using namespace std;

int main(){

    srand( (unsigned) time(NULL) * getpid());


    long double* parameter1 = new long double[4];
    parameter1[0] = 100.;
    parameter1[1] = 1.;
    parameter1[2] = 0.6;
    parameter1[3] = 1.;



    long double mu = asymptoticExpectedValue(parameter1);
    long double sigma = asymptoticStd(parameter1);


    boost::math::normal gaussianApproximation(mu, sigma);

    






    unsigned long int M = 1;
    unsigned long int n = 1;
    unsigned long int m = 0;

    cout<<"Choose x to fix the magnitude of probability to estimate (x>0):  ";
    cin>> m ;


    cout<<"Choose number of estimations for the boxplot: ";
    cin>> n ;

    cout<<endl;

    cout<<"Choose the simulation effort (size of Monte Carlo Sum): ";
    cin>> M ;

    cout<<endl;



    boost::math::normal gaussianApproximation1(0, 1);
    

    long double a = quantile(gaussianApproximation, 1 - 1./pow(10,m));




    vector<long double> Estimations;
    long double estimation = 0.0;
    vector<long double> jumpTimes;


    long double* parameter2 = new long double[4];
    parameter2[0] = parameter1[0];
    parameter2[1] = getLambda0(parameter1[0],parameter1[2],parameter1[3],a);
    parameter2[2] = parameter1[2];
    parameter2[3] = parameter1[3];









    for(int k=0; k<n; k++){

        for(int i=0; i<M ; i++){

            jumpTimes = Ogata(parameter2);
            estimation = estimation + (jumpTimes.size()>a)*likely_Hood(parameter1,parameter2,jumpTimes);

        }


        estimation = estimation/M;

        Estimations.push_back(estimation);

        cout<<100*(k+1)/(double)n<<" % done ......    ";
        cout<<Estimations.size()<<endl;
  }


  save_vector("estimations.txt",Estimations);
  
  

}
