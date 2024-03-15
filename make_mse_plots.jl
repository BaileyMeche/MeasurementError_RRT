using PyPlot

struct MixtureModel
    p
    q
    num_trials
    num_participants
    pix
    piy
    c
    A
end

function compute_Py(model::MixtureModel)
    q_term = model.q * (1 - model.c + model.A*(-1 + 2*model.c)*model.pix)
    p_term = model.p * (model.c + model.A*model.pix - 2*model.A*model.c*model.pix)
    qp1_term = (1 - model.p - model.q) * (model.c + model.piy - 2*model.c*model.piy) 
    return q_term + p_term + qp1_term
end

function compute_mse_m(model::MixtureModel)
    Pyc = compute_Py(model)
    return (Pyc * (1-Pyc)) / (model.num_participants * (1 - 2*model.q + 2*model.piy*(-1 + model.p + model.q))^2)
end

x_axes = [0:0.005:0.3, 0:0.005:0.4, 0:0.005:0.6, 0:0.005:0.7]
y_axes = []

for i in 1:4
    structs = MixtureModel[MixtureModel(last(x_axes[i]), q, 10000, 500, 0, 1/12, c, 1) for q in x_axes[i], c in [0.01, 0.05, 0.10]]
    push!(y_axes, compute_mse_m.(structs))
end

fig, axis = subplots(2, 2, sharex=false, sharey=false, figsize=(10, 6))

xlims = [[0, 0.3], [0, 0.4], [0, 0.6], [0, 0.7]]
ylims = [[0, 0.003], [0, 0.014], [0, 0.5], [0, 0.5]]

for i in 1:2, j in 1:2
    axis[i, j].plot(x_axes[2*(i-1)+j], y_axes[2*(i-1)+j])
    axis[i, j].grid(true)
    axis[i, j].set_xlabel("\$q\$")
    axis[i, j].set_ylabel("MSE" * "\$[\\hat{m}]\$")
    axis[i, j].legend(("\$m=0.01\$", "\$m=0.05\$", "\$m=0.10\$"), loc="upper left")
    axis[i, j].set_xlim(xlims[2*(i-1)+j])
    axis[i, j].set_ylim(ylims[2*(i-1)+j])
end

fig.suptitle("MSE" * "\$[\\hat{m}]\$" * " vs " * "\$q\$")
fig.tight_layout()
savefig("Simulation Output - With Trust/mse_vs_q.png", dpi=600, bbox_inches="tight")
