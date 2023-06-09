# 相关公式

#### 数学公式Latex

$$
\frac{\partial f}{\partial x}=2\sqrt{a}x
$$

$$
y_i=f(b_i+\sum_{i=1}^{n} (x_i\times \omega_{ij}))
$$

$$
\begin{cases}
    x^l_m=b^l_m+\sum_{i=1}^{k} \omega_{im}^{l-1}\times y^{l-1}_i \\
    y^l_m=f(b^l_m+\sum_{i=1}^{k} \omega_{im}^{l-1}\times y^{l-1}_i)=f(x_m^l)
\end{cases}
$$

$$
L_k=\frac{-1}{N}\sum_{xyc}
\begin{cases}
     x=(1-\hat{Y_{xyc}})^\alpha log(\hat{Y_{xyc}}) & 如果 Y_{xyc}=1 \\
     y=(\hat{Y_{xyc}})^\alpha (1-Y_{xyc})^\beta log(1-\hat{Y_{xyc}}) & \ \ \quad 其它
\end{cases}
$$

$$
Y_{x y c}\implies\exp\Big(-\frac{(x-\tilde{p}_{x})^{2}+(y-\tilde{p}_{y})^{2}}{2\sigma_{p}^{2}}\Big)
$$

$$
p_k=\left({\frac{x_{1}^{(k)}+x_{2}^{(k)}}{2}},\;{\frac{y_{1}^{(k)}+y_{2}^{(k)}}{2}}\right)
$$

$$
s_k=(x_{2}^{(k)}-x_{1}^{(k)},y_{2}^{(k)}-y_{1}^{(k)})
$$
$$
L_{size}=\sum_{k=1}^{N}\left | \hat{S}_{pk} -s_k \right | 
$$
$$
L_{off}=\frac{1}{N} \sum_{p} \left | \hat{O}_p -(\frac{p}{R} - p) \right |
$$
