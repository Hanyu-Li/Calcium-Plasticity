/***** LIF static transfer function (white noise inputs) *****/

#include <math.h>
#include <stdio.h>

#define SQPI sqrt(4.*atan(1.))
#define TWOPI (8.*atan(1.))
#define a1  -1.26551223e0
#define a2  1.00002368e0
#define a3  .37409196e0
#define a4  .09678418e0
#define a5  -.18628806e0
#define a6  .27886087e0
#define a7  -1.13520398e0
#define a8 1.48851587e0
#define a9 -.82215223e0
#define a10 .17087277e0
#define NDAT 1000000

/***** Single neuron parameters ****/

#define taum 0.02 /*** membrane time constants in seconds ***/
#define taurp 0.002 /*** refractory period in seconds ***/
#define theta 20. /*** threshold in mV ****/
#define hvr 10. /**** reset in mV ****/


float nerf(float z)           /* function exp(z^2)(erf(z)+1)  */
{
  float t,ef,at,bt;
  float w,fex;
  w = fabs(z);
  t = 1.0e0/(1.0e0 + 0.5e0 * w);
  at=a1+t*(a2+t*(a3+t*(a4+t*(a5+t*(a6+t*(a7+t*(a8+t*(a9+t*a10))))))));
  ef=t*exp(at);
  if(z>0.0e0)
    ef = 2.0e0*exp(w*w)-ef;  
  return(ef);
}

float trans(float x,float y)     /* transfer function 
x=(threshold-mu)/sigma, y=(reset-mu)/sigma */ 
{       
  float w,z,cont,ylow;
  int n,i,N=1000;
  w=0.0e0;
  if(x<-80.&&y<-80.) {
    w=log(y/x)-0.25/pow(x,2.)+0.25/pow(y,2.);
    w=1./(taurp+taum*w);
  }
  else if(y<-80.) {
    ylow=-100.; 
    N=(int)(1000.*(x-ylow));
    for(i=0;i<=N;i++) {
      z=ylow+(x-ylow)*(float)(i)/(float)(N);
      cont=nerf(z);
      if(i==0||i==N) w+=0.5*cont;
      else w+=cont;
    }
    w*=(x-ylow)*SQPI/(float)(N);
    w+=log(-y/100.)-0.000025+0.25/pow(y,2.);
    w=1.0e0/(taurp+taum*w);
  }
  else {
    ylow=y;
    N=(int)(1000.*(x-ylow));
    for(i=0;i<=N;i++) {
      z=ylow+(x-ylow)*(float)(i)/(float)(N);
      cont=nerf(z);
      if(i==0||i==N) w+=0.5*cont;
      else w+=cont;
    }
    w*=(x-ylow)*SQPI/(float)(N);
    w=1.0e0/(taurp+taum*w);
  }
  return(w);
}

float function(float x)
{
  float w;
  float y,ymin=-20.;
  int i,N=10000;
  w=0.0e0;
  for(i=0;i<=N;i++) {       
    y=ymin+(x-ymin)*(float)(i)/(float)(N);
    if(i==0||i==N) w+=0.5*pow(nerf(y)*exp(0.5*(x*x-y*y)),2.);
    else w+=pow(nerf(y)*exp(0.5*(x*x-y*y)),2.);
  }
  w*=(x-ymin)/(float)(N);
  return(w);
}

float cv(float x,float y)     /* coefficient of variation  
x=(threshold-mu)/sigma, y=(reset-mu)/sigma */ 
{       
  float w,z,v,cont,zmin,ylow;
  int n,i,j,N=10000;
  w=0.0e0;
  if(x<-100.&&y<-100.) {
    w=0.5/pow(x,2.)-0.5/pow(y,2.);
  }
  else if(y<-100.) {
    ylow=-100; 
    N=(int)(100.*(x-ylow));
    for(i=0;i<=N;i++) {
      z=ylow+(x-ylow)*(float)(i)/(float)(N);
      if(z>-10.) cont=function(z);
      else cont=1./(-TWOPI*pow(z,3.))+1.5/(TWOPI*pow(z,5.));
      if(i==0||i==N) w+=0.5*cont;
      else w+=cont;
    }
    w*=(x-ylow)*TWOPI/(float)(N);
    w+=0.00005-0.5/pow(y,2.);
  }
  else {
    ylow=y;
    N=(int)(100.*(x-ylow));
    for(i=0;i<=N;i++) {
      z=ylow+(x-ylow)*(float)(i)/(float)(N);
      if(z>-10.) cont=function(z);
      else cont=1./(-TWOPI*pow(z,3.))+1.5/(TWOPI*pow(z,5.));
      if(i==0||i==N) w+=0.5*cont;
      else w+=cont;
    }
    w*=(x-ylow)*TWOPI/(float)(N);
  }
  w*=pow(trans(x,y)*taum,2.);
  return(sqrt(w));
}


main() { 
  float mu,sigma,dmu;
  sigma=0.1;
  dmu=0.1;
  for(mu=0.;mu<40.;mu+=dmu) printf("%f %f\n",mu,trans((theta-mu)/sigma,(hvr-mu)/sigma));
}


