under the dynamics:
\begin{equation}
\begin{aligned}
\dot{\theta} & =\omega \\
\dot{\omega} & =-\nu \omega+\theta+\dot{v} \\
\dot{v} & =\frac{1}{\tau}(u(t)-v(t)) .
\end{aligned}
\end{equation}



\begin{equation}
\begin{aligned}
& \begin{aligned}
H & =\theta^2+p_1 w+p_2\left(-b w-w_2^2 \theta-\frac{x_0^1}{p}\right) \\
& =L-p_1 \dot{1}-p_2 \dot{w}
\end{aligned} \\
& \operatorname{Si}_2 P_2<0 \Rightarrow x_x^{\prime \prime}=C \\
& x p_2>0 \quad x=\ldots \Gamma \text {. } \\
&
\end{aligned}
\end{equation}

The co-state equations for this system are :
$$
\begin{aligned}
& \dot{p}_1=-\partial_\theta H=-\omega_0^2 p_2 \\
& \dot{p}_2=-\partial_\omega H=-2 \omega-p_1+\xi \omega_0 p_2 \\
& \dot{p}_3=-\partial_\phi H=-\omega_0^2 \lambda p_2+p_3 \frac{\Omega}{\Delta} \cosh ^{-2} \frac{\phi_c-\phi}{\Delta}
\end{aligned}
$$
Following the Pontryagin seminal idea, the value of the control parameter $\phi_c$ can be chosen in order to maximize the Hamiltonian value. The bang-bang controller will be optimal if the fourth term in the Hamiltonian (11) which contains $\phi_c$ is linear, which is