
# Ex 2  
## Q1  
Given an agent with varying location and speed, where both are in one dimesion only.  
The movement model is a constant speed model.  
The initial location error is $2$ meters, and the initial speed error is $1.2$ feet($1$ feet = $0.3048$ meter).  
The sensor measures only the location, and its accuracy is a Gaussian distribution with a variance of $0.5$ feet.  
The initial state is $x = 8, v = 5\frac{m}{s}$
$\Delta t = 1~ sec$
  
1. Write the following matrices:  
    $H,P,F$ and the Kalman Gain ($K$)  
1. Assuming the sensor measures that the agent is at $43$ feet ($13.1054$ meter),   
    calculate the status vectors ($P,X$) and the new $K$ after the update.  
      
### A1  
1. $P = \begin{bmatrix}  
4 & 0\\
0& 0.36576^2\\
\end{bmatrix}$
$F = \begin{bmatrix}  1 & \Delta t \\0 & 1 \\\end{bmatrix}=
\begin{bmatrix}  1 & 1\\0 & 1 \\\end{bmatrix}$
$H = \begin{bmatrix}  
1 & 0\\
\end{bmatrix}$
$K =HP_0H^T(HP_0H^T-R_0)^{-1}=\\4\cdot(4+0.1524^2)^{-1}=
\frac{4}{4.0232}=0.9942$
1. Measurement $(\mu_1,\Sigma_1)=(Z_1,R_1) = (13.1054,0.1524^2)$ 
$$\hat x=x_1+K(\mu_1-\mu_0)=\\
8+K(13.1054-8)=\\
8+0.9942(5.1054) = \\8+5.076=13.076$$
$$\hat P=F\cdot P\cdot F^T=\\
\begin{bmatrix}  1 & 1\\0 & 1 \\\end{bmatrix}
\begin{bmatrix}  
4 &0.73152\\
0.73152& 0.13378\\
\end{bmatrix}
\begin{bmatrix} 1 & 0\\1  & 1 \\\end{bmatrix}=\\
\begin{bmatrix}  1 & 1\\0 & 1 \\\end{bmatrix}
\begin{bmatrix}  
4.73152&0.73152\\
0.73152&  0.13378\\
\end{bmatrix}
=\\
\begin{bmatrix}  
5.4630& 0.7025\\
0.7025&  0.8653\\
\end{bmatrix}$$

$$
K_1 = HP_kH_T(H_KP_KH_K^T+R_K)^{-1}=\\
\begin{bmatrix}1&0\end{bmatrix}
\begin{bmatrix}  
5.4630& 0.7025\\
0.7025&  0.8653\\
\end{bmatrix}
\begin{bmatrix}1\\0\end{bmatrix}
\left ( \begin{bmatrix}1&0\end{bmatrix}
\begin{bmatrix}  
5.4630& 0.7025\\
0.7025&  0.8653\\
\end{bmatrix}
\begin{bmatrix}1\\0\end{bmatrix}+0.15242^2 \right)^{-1}=\\
\frac{5.4630}{5.4630+0.02323}=
0.99576
$$

## Q2
Assume that the sensor measures both location(feet) and speed ($\frac{m}{s}$). The variance of the sensor location is $0.5 ~ feet$ and the variance for the sensor speed measurement is $4\frac{m}{s}$.
1. Repeat [Q1] , what will be the shape of the Kalman Gain matrix?
2. Assuming that the sensor measured at $\Delta t$ a location of $43 ~feet$ and a speed of $4\frac{m}{s}$, compute the status vector ($P,X$), and the Kalman gain after the **Update**.

### A2
1. $F = 
\begin{bmatrix}  
1 & 1\\
0 & 1 \\
\end{bmatrix}$
$P_0 = \begin{bmatrix}  
4 &0\\
0&0.36576^2\\
\end{bmatrix}$
$P' = FP_0F^T\\
=\begin{bmatrix}  1 & 1\\0 & 1 \\\end{bmatrix}
\begin{bmatrix}  
4 & 0\\
0&0.13378\\
\end{bmatrix}
\begin{bmatrix}  1 & 0\\1 & 1 \\\end{bmatrix}\\
=\begin{bmatrix}
4& 0 \\
0& 0.13378
\end{bmatrix}$
$H = \begin{bmatrix}  
1 & 0\\
0 & 1
\end{bmatrix}$
$K =HP'H^T(HP'H^T-R_0)^{-1}=\\
\begin{bmatrix}
1&0\\0&1
\end{bmatrix}
\begin{bmatrix}
4& 0 \\
0& 0.13378
\end{bmatrix}
\begin{bmatrix}1&0\\0&1\end{bmatrix}
\left (
\begin{bmatrix}1&0\\0&1\end{bmatrix}
\begin{bmatrix}
4& 0 \\
0& 0.13378
\end{bmatrix}
\begin{bmatrix}1&0\\0&1\end{bmatrix} +
\begin{bmatrix}
0.1524^2&0\\
0& 4^2
\end{bmatrix}
\right )^{-1}\\=
\begin{bmatrix}
4& 0 \\
0& 0.13378
\end{bmatrix}\left (
\begin{bmatrix}
4& 0 \\
0& 0.13378
\end{bmatrix} + 
\begin{bmatrix}
0.1524^2&0\\
0& 4^2
\end{bmatrix}
\right )^{-1}=\\
\begin{bmatrix}
4& 0 \\
0& 0.13378
\end{bmatrix}
\begin{bmatrix}  
5.62004576&  0\\
0  & 16.13378
\end{bmatrix}^{-1}\\=
\begin{bmatrix}
9.942\cdot 10^{-1}& 4.894\cdot 10^{-3}\\
4.894\cdot 10^{-3}& 8.479\cdot 10^{-1}
\end{bmatrix}$
Thus, $K$ is a $2\times 2$ shape matrix.

2. Measurement $\begin{bmatrix}13.1064\\4\end{bmatrix}$ 
$\hat x^- = Fx_0 = \begin{bmatrix}1&1\\0&1\end{bmatrix}\begin{bmatrix}8\\5\end{bmatrix} = \begin{bmatrix}13\\5\end{bmatrix}$

$$\hat x=\begin{bmatrix}13\\5\end{bmatrix}+K(\begin{bmatrix}13.1064\\4\end{bmatrix}-\begin{bmatrix}13\\5\end{bmatrix})=\\
\begin{bmatrix}13\\5\end{bmatrix}+
\begin{bmatrix}
9.942\cdot 10^{-1}& 4.894\cdot 10^{-3}\\
4.894\cdot 10^{-3}& 8.479\cdot 10^{-1}
\end{bmatrix}
\begin{bmatrix}0.1064\\-1\end{bmatrix}=\\
\begin{bmatrix}13\\5\end{bmatrix}+
\begin{bmatrix}0.10089\\-0.8473\end{bmatrix}=\begin{bmatrix}13.10089\\4.1526\end{bmatrix}$$
$$
P_1 = (I-KH)P'\\=
\left (I - \begin{bmatrix}
9.942\cdot 10^{-1}& 4.894\cdot 10^{-3}\\
4.894\cdot 10^{-3}& 8.479\cdot 10^{-1}
\end{bmatrix}
\begin{bmatrix}  
1 & 0\\
0 & 1
\end{bmatrix}\right )
\begin{bmatrix}  
4 & 0\\
0&0.13378
\end{bmatrix}\\
=\begin{bmatrix}  
0.00574466& 0.99510514\\
0.99510514& 0.15209976
\end{bmatrix}
\begin{bmatrix}  
4 & 0\\
0&0.13378
\end{bmatrix}\\
=\begin{bmatrix}  
0.15687271& 0.13389406]\\
4.13389406& 0.15347351
\end{bmatrix}
$$

## Q3
 Write a program in python that performs Kalman filter as described this question (1-D location and speed), and returns $P~and~X$
