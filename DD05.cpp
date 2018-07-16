#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
/* DD05 potential */

#include "spilady.h"

double bigf_gen(double rho){

    double aa = 4.100199340884814;
    double bb = 1.565647547483517;

    double bigf = 0e0;
    if (rho >= 1e0){
        bigf = -aa*sqrt(rho);
    } else {
        bigf = -aa*sqrt(rho)-bb/log(2e0)*(1e0-sqrt(rho))*log(2e0-rho);
    }
    return bigf;
}

double smallf_gen(double rij){

    const int N = 6;
    double a[N], r[N];

    a[0] =  0.9332056681088162;
    a[1] = -1.162558782567700;
    a[2] = -0.3502026949249225;
    a[3] =  0.4287820835430028;
    a[4] =  4.907925057809273;
    a[5] = -5.307049068415304;
    r[0] =  3.0;
    r[1] =  2.8666666666666666;
    r[2] =  2.7333333333333333;
    r[3] =  2.6;
    r[4] =  2.4;
    r[5] =  2.3;

    double smallf = 0e0;
    for (int i = 0; i < N; ++i)
        if (rij < r[i])  smallf += a[i]*pow(r[i]-rij,3);

    return smallf;
}

double pairZBL(double rij){

   double bb0 =  0.1818e0;
   double bb1 =  0.5099e0;
   double bb2 =  0.2802e0;
   double bb3 =  0.02817e0;

   double nzz = 26e0;       //iron
   double abohr = 0.52917720859e0;
   double as = 0.88535e0*abohr/(2e0*pow(nzz,0.23));

   double xx = rij/as;

   double pair = bb0*exp(-3.2*xx)+bb1*exp(-0.9423*xx)+bb2*exp(-0.4029*xx)+bb3*exp(-0.2016*xx);
   pair = nzz*nzz*14.3992*pair/rij;

   return pair;
}

double pair_gen(double rij){

    const int N = 8;
    double V[N], r[N];

    V[0] = -0.1960674387419232;
    V[1] =  0.3687525935422963;
    V[2] = -1.505333614924853;
    V[3] =  4.948907078156191;
    V[4] = -4.894613262753399;
    V[5] =  3.468897724782442;
    V[6] = -1.792218099820337;
    V[7] = 80.22069592246987;
    r[0] = 4.1;
    r[1] = 3.8;
    r[2] = 3.5;
    r[3] = 3.2;
    r[4] = 2.9;
    r[5] = 2.6;
    r[6] = 2.4;
    r[7] = 2.3;

    double pair = 0e0;

    double x1 = 1.8;       // spline fit start point
    double x2 = 2.0;       // spline fit end point

    double offset = 0e0;

    if (rij <= x1){

        pair = pairZBL(rij) + offset;

    } else if (rij <= x2) {
        double y1 = pairZBL(x1) + offset;
        double Dy1 = (pairZBL(x1+1e-7)-pairZBL(x1-1e-7))/2e-7;
        double  D2y1 =(pairZBL(x1+1e-7)+pairZBL(x1-1e-7)-2e0*pairZBL(x1))/pow(1e-7,2);

        double y2 = 0e0;
        for (int i = 0; i < N ; ++i)
            if(x2 < r[i]) y2 += V[i]*pow((r[i]-x2),3);

        double Dy2 = 0e0;
        for (int i = 0; i < N ; ++i)
            if(x2 < r[i]) Dy2 -= 3e0*V[i]*pow((r[i]-x2),2);

        double D2y2 = 0e0;
        for (int i = 0; i < N ; ++i)
            if(x2 < r[i]) D2y2 += 6e0*V[i]*(r[i]-x2);

        double a5 = (-6e0*Dy1*x1-6e0*Dy2*x1+D2y1*pow(x1,2)-D2y2*pow(x1,2)
                   +6e0*Dy1*x2+6e0*Dy2*x2-2e0*D2y1*x1*x2+2e0*D2y2*x1*x2
                   +D2y1*pow(x2,2)-D2y2*pow(x2,2)+12*y1-12*y2)
                   /(2e0*pow((x1-x2),5));
        double a4 = (-2e0*D2y1*pow(x1,3)+3e0*D2y2*pow(x1,3)+D2y1*pow(x1,2)*x2
                   -4e0*D2y2*pow(x1,2)*x2+4*D2y1*x1*pow(x2,2)-D2y2*x1*pow(x2,2)
                   -3e0*D2y1*pow(x2,3)+2e0*D2y2*pow(x2,3)
                   +2e0*Dy1*(7e0*pow(x1,2)+x1*x2-8e0*pow(x2,2))
                   +2e0*Dy2*(8e0*pow(x1,2)-x1*x2-7e0*pow(x2,2))
                   -30e0*x1*y1-30e0*x2*y1+30e0*x1*y2+30e0*x2*y2)
                   /(2e0*pow((x1-x2),5));
        double a3 = -(-D2y1*pow(x1,4)+3e0*D2y2*pow(x1,4)-4e0*D2y1*pow(x1,3)*x2
                   +8e0*D2y1*pow(x1,2)*pow(x2,2)-8e0*D2y2*pow(x1,2)*pow(x2,2)
                   +4e0*D2y2*x1*pow(x2,3)-3e0*D2y1*pow(x2,4)+D2y2*pow(x2,4)
                   +4e0*Dy1*(2e0*pow(x1,3)+8e0*pow(x1,2)*x2-7e0*x1*pow(x2,2)-3e0*pow(x2,3))
                   +4e0*Dy2*(3e0*pow(x1,3)+7e0*pow(x1,2)*x2-8e0*x1*pow(x2,2)-2e0*pow(x2,3))
                   -20e0*pow(x1,2)*y1-80e0*x1*x2*y1-20e0*pow(x2,2)*y1
                   +20e0*pow(x1,2)*y2+80e0*x1*x2*y2+20e0*pow(x2,2)*y2)
                   /(2e0*pow((x1-x2),5));
        double a2 = (D2y2*x1*pow((x1-x2),2)*(pow(x1,2)+6e0*x1*x2+3e0*pow(x2,2))
                   -x2*(3e0*D2y1*pow(x1,4)-8e0*D2y1*pow(x1,2)*pow(x2,2)
                   +4e0*D2y1*x1*pow(x2,3)+D2y1*pow(x2,4)
                   -12e0*Dy1*x1*(2e0*pow(x1,2)+x1*x2-3e0*pow(x2,2))
                   +12e0*Dy2*x1*(-3e0*pow(x1,2)+x1*x2+2e0*pow(x2,2))
                   +60e0*pow(x1,2)*y1+60e0*x1*x2*y1
                   -60e0*pow(x1,2)*y2-60e0*x1*x2*y2))
                   /(2e0*pow((x1-x2),5));
        double a1 = (2e0*Dy2*pow(x1,2)*(pow(x1,3)-5e0*pow(x1,2)*x2-8e0*x1*pow(x2,2)+12e0*pow(x2,3))
                  +x2*(-D2y2*pow(x1,2)*pow((x1-x2),2)*(2e0*x1+3e0*x2)
                  +x2*(-2e0*Dy1*(12e0*pow(x1,3)-8e0*pow(x1,2)*x2-5e0*x1*pow(x2,2)+pow(x2,3))
                  +x1*(D2y1*pow((x1-x2),2)*(3e0*x1+2e0*x2)+60e0*x1*(y1-y2)))))
                  /(2e0*pow((x1-x2),5));
        double a0 = (D2y2*pow(x1,3)*pow((x1-x2),2)*pow(x2,2)+8e0*Dy1*pow(x1,3)*pow(x2,3)
                   -D2y1*pow(x1,4)*pow(x2,3)-10e0*Dy1*pow(x1,2)*pow(x2,4)
                   +2e0*D2y1*pow(x1,3)*pow(x2,4)+2e0*Dy1*x1*pow(x2,5)-D2y1*pow(x1,2)*pow(x2,5)
                   -2e0*Dy2*pow(x1,3)*x2*(pow(x1,2)-5e0*x1*x2+4e0*pow(x2,2))
                   -20e0*pow(x1,2)*pow(x2,3)*y1+10e0*x1*pow(x2,4)*y1-2e0*pow(x2,5)*y1
                   +2e0*pow(x1,5)*y2-10e0*pow(x1,4)*x2*y2+20e0*pow(x1,3)*pow(x2,2)*y2)
                   /(2e0*pow((x1-x2),5));

        pair = a5*pow(rij,5)+a4*pow(rij,4)+a3*pow(rij,3)+a2*pow(rij,2)+a1*rij+a0;

    } else {

        for (int i = 0; i < N; ++i)
            if (rij < r[i]) pair += V[i]*pow((r[i]-rij),3);

    }
    return pair;

}

#endif
