(*Mathematica Code for MLP*)
(*Normal random variables with variance 1*)
xrv:=2{Random[],Random[]}-1;
gauss:=Module[{t},t=xrv;t=If[t.t>=1,gauss,t[[1]]Sqrt[-2Log[t.t]/t.t]]]
noise=Table[.5{gauss,gauss},{i,1000}]; (*Normal noise var=0.25*)
(*Logistic function:*)
f[x_]=1/(1+Exp[-x]) (*works on vectors component-wise*)
(*For the XOR problem with n hidden neurons*)
(*Defining weight matrices in-hidden:*)
intoh[n_]:=Table[4.Random[]-2.,{n},{3}]
(*and hidden-op:*)
htout[n_]:=Table[4. Random[]-2.,{n+1}] (*extra one for offset*)
(*For the 2-hidden unit network:*)
(*First set up weight matrices:*)
weight1=intoh[2]; (*for 2 hidden units*)
weight2=htout[2];
(*Then a table for the errors:*)
etab=Table[0,{1000}];
(*and an offset vector:*)
ones=Table[-1,{16}];
(*Target output*)
trainop={0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0};
(*Full set of training vectors*)
trainset=<<"training.dat";
(*Training patterns are 3*16 matrix; last row is offset:*)


train={{-0.310691, -0.309003, 1.25774, 1.31959, -0.0897083, -0.457115,
1.42524, 1.43962, -0.21377, -0.16744, 0.579612, 1.90558, 0.442017,
0.204012, 1.75664, 0.584128},
{0.0164278, 0.898471, -0.231735, 0.82952, -1.02045, 1.84369, 0.111823,
0.28365, 0.0759174, 0.985518, 0.584378, 0.434351, 0.35245, -0.0194183,
-0.336488, 1.45608}, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
-1, -1, -1, -1}};
(*Then the training loop of 1000 epochs:*)
Do[hid=f[weight1.train]; (*hidden outputs*)
out=f[weight2.Join[hid,ones]]; (*net output*)
e=trainop-out; (*output error*)
etab[[i]]=e.e; (*squared error on epoch i*)
deltaout=e*out(1-out); (*delta for output unit*)
ehid=Take[Outer[Times,weight2,deltaout],2]; (*backpropagate,strip offset*)
deltahid=ehid*hid*(1-hid); (*get delta for hidden layer*)
weight2+=deltaout.Transpose[Join[hid,ones]]; (*update output weight matrix*)
weight1+=deltahid.Transpose[train];
If[Mod[i,100]==0,Print[etab[[i]]],0],{i,1000}]
