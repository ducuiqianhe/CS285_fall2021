import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 10, 1)
y_ant_train = [4713.6, 708.3, 3955.44, 4224.5, 4354.77, 4588.86, 4360.04, 4461.08, 4815.14, 4650.57]
y_ant_eval = [814.95, 3857.96, 4230.64, 4496.68, 4528.17, 4574.2, 4597.87, 4756.13, 4812.55, 4708.86]
y_ant_loss = [153.86, 587.92, 126.81, 82.63, 76.31, 65.96, 83.84, 103.63, 111.23, 66.64]

y_hopper_train = [3772.67, 374.48, 1097.05, 1906.85, 3562.31, 3271.99, 1878.26, 3786.11, 3765.8, 3782.5]
y_hopper_eval = [442.28, 1047.53, 2124.09, 3572.16, 3149.38, 2267.65, 3754.51, 3766.08, 3787.28, 3774.29]
y_hopper_loss = [254.17, 153.35, 651.29, 93.55, 771.43, 577.35, 43.89, 4.36, 2.92, 6.07]


# plt.xlabel('num_agent_train_steps_per_iter')
# plt.ylabel('Eval_AverageReturn')
# plt.plot(x, y, 'b')


# plt.xlabel('n_iter')
# plt.ylabel('performance')
# plt.plot(x, y_ant_train, 'b', x, y_ant_eval, "r")
# plt.bar(x, height=y_ant_loss, width=0.3)

fig, ax1 = plt.subplots()


color = 'tab:blue'
ax1.set_xlabel('n_iter')
ax1.set_ylabel('mean performance', color=color)
ax1.plot(x, y_hopper_train, 'g', label="train_mean")
ax1.plot(x, y_hopper_eval, 'b', label="eval_mean")
ax1.axhline(y=3772, color='r', label="expert mean", linestyle='-')
ax1.axhline(y=442, color='y', label="behavior cloning", linestyle='-')
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc="right")

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('eval loss', color=color)  # we already handled the x-label with ax1
ax2.bar(x, height=y_hopper_loss, width=0.3, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Hopper DAgger")
fig.tight_layout()


# t = np.arange(0., 5., 0.2)
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r')
#
# plt.bar([1, 2, 3, 4], height=[1, 4, 9, 16], width=0.3)


# plt.plot(t, t, 'b--', t, t**2, 'bs', t, t**3, 'g^')
# plt.ylabel('some numbers')
plt.show()