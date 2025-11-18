+++
date = '2025-11-18T20:37:29+08:00'
draft = false
title = 'Code of DDPM and DDIM'
+++

# DDPM
## 公式
✅ 公式 (15)
$$
x_0 \approx \hat{x}_0 
= \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t)}
       {\sqrt{\bar{\alpha}_t}}
\tag{15}
$$

✅ 公式 (6)
$$
q(x_{t-1} \mid x_t, x_0)
= \mathcal{N}\bigl(x_{t-1};\; \mu_t(x_t, x_0),\; \tilde{\beta}_t I\bigr)
\tag{6}
$$
✅ 公式 (7)
$$
\mu_t(x_t, x_0)
:= 
\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\, x_0
+
\frac{\sqrt{\bar{\alpha}_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t,
\quad
\tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
\tag{7}
$$
## Code

1.计算 $\bar{\alpha}_t, \bar{\alpha}_{t-1}, \alpha_t$

```python
t = timestep

prev_t = self.previous_timestep(t)

# 1. compute alphas, betas
alpha_prod_t = self.alphas_cumprod[t]
alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
beta_prod_t = 1 - alpha_prod_t
beta_prod_t_prev = 1 - alpha_prod_t_prev
current_alpha_t = alpha_prod_t / alpha_prod_t_prev
current_beta_t = 1 - current_alpha_t
```
2.计算$$
x_0 \approx \hat{x}_0 
= \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t)}
       {\sqrt{\bar{\alpha}_t}}
\tag{15}
$$
```python
	# 2. compute predicted original sample from predicted noise also called
	# "predicted x_0" of formula (15) from https://huggingface.co/papers/2006.11239

pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

```
3.计算公式（7）里的$x_0和x_t前边的系数$  $\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}$ 和 $\frac{\sqrt{\bar{\alpha}_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$
```python
# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
# See formula (7) from https://huggingface.co/papers/2006.11239
pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

```
4.计算公式（7）的 $\mu_t(x_t, x_0)$ 
$$\mu_t(x_t, x_0)
:= 
\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\, x_0
+
\frac{\sqrt{\bar{\alpha}_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t,$$

```python
# 5. Compute predicted previous sample µ_t
# See formula (7) from https://huggingface.co/papers/2006.11239
pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
```
4.计算公式（7）中的方差 $\tilde{\beta}_t$  
$$\tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
\tag{7}$$
```python
prev_t = self.previous_timestep(t)

alpha_prod_t = self.alphas_cumprod[t]
alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

# For t > 0, compute predicted variance βt (see formula (6) and (7) from https://huggingface.co/papers/2006.11239)
# and sample from it to get previous sample
# x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

```
7.采样$x_{t-1} = \mu_t(x_t, x_0) + \tilde{\beta}_t * I$
```python
variance_noise = randn_tensor(
	model_output.shape, generator=generator, device=device, dtype=model_output.dtype)

 
pred_prev_sample = pred_prev_sample + variance * variance_noise
```


# DDIM
https://huggingface.co/docs/diffusers/v0.35.1/en/api/schedulers/ddim#diffusers.DDIMScheduler

https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py#L342

https://arxiv.org/abs/2010.02502

https://huggingface.co/papers/2010.02502
## **DDIM 核心（无随机项）公式**

$$
x_{t-1}
= \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0+\sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t),
$$

其中

$$
\hat{x}_0
= \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}
       {\sqrt{\bar{\alpha}_t}}.
$$

## **DDIM 核心（含随机项）公式**
$$
x_{t-1}
= \sqrt{\alpha_{t-1}}
\left(
    \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}
         {\sqrt{\alpha_t}}
\right)+\sqrt{1-\alpha_{t-1}-\sigma_t^{2}}\;\epsilon_\theta^{(t)}(x_t)+\sigma_t \epsilon_t
$$
## DDIM Code
DDIM定义
```python
# See formulas (12) and (16) of DDIM paper https://huggingface.co/papers/2010.02502
# Ideally, read DDIM paper in-detail understanding

# Notation (<variable name> -> <name in paper>
# - pred_noise_t -> e_theta(x_t, t)
# - pred_original_sample -> f_theta(x_t, t) or x_0
# - std_dev_t -> sigma_t
# - eta -> η
# - pred_sample_direction -> "direction pointing to x_t"
# - pred_prev_sample -> "x_t-1"
```
1.计算下一步时间步t-1
```python
# 1. get previous step value (=t-1)
# 999, 979, 959, ...
prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
```
2.计算$\alpha_t,\alpha_{t-1},\beta_t$
```python
# 2. compute alphas, betas
# 计算\alpha_t
alpha_prod_t = self.alphas_cumprod[timestep]
# 计算\alpha_{t-1}
alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

# 计算 \beta,即1-\alpha_t。
beta_prod_t = 1 - alpha_prod_t
```
3.计算 pred_original_sample=$\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}$
```python
# 3. compute predicted original sample from predicted noise also called
# "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
# sample是x_t，
# 计算    \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}
pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
pred_epsilon = model_output
```
4.计算std_dev_t = $\sigma_t$
```python
# 5. compute variance: "sigma_t(η)" -> see formula (16)
# σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
variance = self._get_variance(timestep, prev_timestep)
std_dev_t = eta * variance ** (0.5)
```
5.计算pred_sample_direction = $\sqrt{1-\alpha_{t-1}-\sigma_t^{2}}\;\epsilon_\theta^{(t)}(x_t)$
```python
# 6. compute "direction pointing to x_t" of formula (12) from https://huggingface.co/papers/2010.02502
pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
```
6.计算$$
x_{t-1}
= \sqrt{\alpha_{t-1}}
\left(
    \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}
         {\sqrt{\alpha_t}}
\right)+\sqrt{1-\alpha_{t-1}-\sigma_t^{2}}\;\epsilon_\theta^{(t)}(x_t)
$$

```python
# 7. compute x_t without "random noise" of formula (12) from https://huggingface.co/papers/2010.02502
prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

```

7.计算
$$
x_{t-1}
= \sqrt{\alpha_{t-1}}
\left(
    \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}
         {\sqrt{\alpha_t}}
\right)+\sqrt{1-\alpha_{t-1}-\sigma_t^{2}}\;\epsilon_\theta^{(t)}(x_t)+\sigma_t \epsilon_t
$$

```python
# \sigma_t \epsilon_t
variance = std_dev_t * variance_noise

prev_sample = prev_sample + variance
```
DDIM的使用
```python

# set step values
# 999, 979, 959, ...
timesteps = (
        np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
        .round()[::-1]
        .copy()
        .astype(np.int64)
            )
# self.scheduler.timesteps就是timesteps
for t in self.progress_bar(self.scheduler.timesteps):
	# 1. predict noise model_output
	model_output = self.unet(image, t).sample

	# 2. predict previous mean of image x_t-1 and add variance depending on eta
	# eta corresponds to η in paper and should be between [0, 1]
	# do x_t -> x_t-1
	image = self.scheduler.step(
		model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
	).prev_sample


```