%NProcShared=2
#n B3LYP/aug-cc-pVDZ EmpiricalDispersion=GD3 int=(grid(ultrafinegrid)) scrf=(PCM,solvent=1,1,1-TriChloroEthane) opt freq=(temperature=298.15,Numerical,Raman,HPModes) 

testing 1 2 3
testing 1 2 3

0 1
C     -4.193114    -0.072755    -0.001305
C     -4.183770    -1.459892    -0.000651
C     -2.996564     0.628970    -0.000914
C     -1.790400    -0.056565     0.000130
C     -1.781056    -1.443702     0.000785
C     -2.977606    -2.145426     0.000394
H     -2.970866    -3.228298     0.000905
H     -0.839523    -1.978798     0.001600
H     -5.117669    -2.008046    -0.000954
H     -5.134647     0.462342    -0.002120
H     -0.856501     0.491589     0.000434
H     -3.003304     1.711842    -0.001425





