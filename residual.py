import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap, grad, config, debug
from jax.flatten_util import ravel_pytree
from utils import *
config.update("jax_enable_x64", True)

class Residual:
    @staticmethod
    def _denormalized_derivatives(model, x, t):
        x = jnp.asarray(x, dtype=jnp.float64)
        t = jnp.asarray(t, dtype=jnp.float64)

        phi_fn = lambda x_, t_: model(x_, t_)[0]
        c_fn = lambda x_, t_: model(x_, t_)[1]
        phi, c = model(x, t)

        # Compute gradients (rescaled)
        phi_t = vmap(grad(phi_fn, argnums=1))(x, t) * t_coef
        phi_x = vmap(grad(phi_fn, argnums=0))(x, t) * x_coef
        phi_xx = vmap(grad(grad(phi_fn, argnums=0), argnums=0))(x, t) * x_coef**2

        c_t = vmap(grad(c_fn, argnums=1))(x, t) * t_coef
        c_x = vmap(grad(c_fn, argnums=0))(x, t) * x_coef
        c_xx = vmap(grad(grad(c_fn, argnums=0), argnums=0))(x, t) * x_coef**2

        return {
            "phi": jnp.asarray(phi, dtype=jnp.float64),
            "c": jnp.asarray(c, dtype=jnp.float64),
            "phi_t": phi_t,
            "phi_x": phi_x,
            "phi_xx": phi_xx,
            "c_t": c_t,
            "c_x": c_x,
            "c_xx": c_xx
        }

    @staticmethod
    def get_res_PHYS(model, x, t):
        derivs = Residual._denormalized_derivatives(model, x, t)
        h_val = h(derivs["phi"])
        h_phi = h_p(derivs["phi"])
        h_xx = h_pp(derivs["phi"]) * derivs["phi_x"]**2 + h_p(derivs["phi"]) * derivs["phi_xx"]
        g_phi = g_p(derivs["phi"])

        res_phi = (
            derivs["phi_t"]
            - 2.0 * A * L * (derivs["c"] - h_val * dc - c_l) * dc * h_phi
            + L * omega_phi * g_phi
            - L * alpha_phi * derivs["phi_xx"]
        )
        res_c = (
            derivs["c_t"]
            - 2.0 * A * M * derivs["c_xx"]
            + 2.0 * A * M * dc * h_xx
        )

        return jnp.concatenate([
            jnp.asarray(res_phi, dtype=jnp.float64),
            jnp.asarray(res_c, dtype=jnp.float64)
        ])

    @staticmethod
    def get_res_IC(model, x, t):
        x = jnp.asarray(x, dtype=jnp.float64)
        t = jnp.asarray(t, dtype=jnp.float64)
        phi_pred, c_pred = model(x, t)
        phi_ic = compute_phi_ic(x, t)
        c_ic = compute_c_ic(x, t)
        return jnp.concatenate([
            phi_pred - phi_ic,
            c_pred - c_ic
        ])

    @staticmethod
    def get_res_BC(model, x, t):
        x = jnp.asarray(x, dtype=jnp.float64)
        t = jnp.asarray(t, dtype=jnp.float64)
        phi_pred, c_pred = model(x, t)
        phi_bc = compute_phi_bc(x, t)
        c_bc = compute_c_bc(x, t)
        return jnp.concatenate([
            phi_pred - phi_bc,
            c_pred - c_bc
        ])

    @staticmethod
    def compute_loss(model, x, t):
        res_phys = Residual.get_res_PHYS(model, x.getval('colloc', 'adapt'), t.getval('colloc', 'adapt'))
        N_phys = res_phys.shape[0] // 2
        loss_AC = jnp.mean(res_phys[:N_phys]**2)
        loss_CH = jnp.mean(res_phys[N_phys:]**2)
        loss_IC = jnp.mean(Residual.get_res_IC(model, x.getval('ic'), t.getval('ic'))**2)
        loss_BC = jnp.mean(Residual.get_res_BC(model, x.getval('bc'), t.getval('bc'))**2)
        return (
            jnp.asarray(loss_AC, dtype=jnp.float64),
            jnp.asarray(loss_CH, dtype=jnp.float64),
            jnp.asarray(loss_IC, dtype=jnp.float64),
            jnp.asarray(loss_BC, dtype=jnp.float64)
        )

    @staticmethod
    def get_criterion(model, x_norm, t_norm, n_adapt, which_criterion="residual"):
        """Select top-k unique indices from residuals or gradients (JAX-jit safe)."""
        if which_criterion == "grad":
            derivs = Residual._denormalized_derivatives(model, x_norm, t_norm)
            criterion_lsts = jnp.asarray(
                [jnp.abs(derivs[k]) for k in ["phi_t", "c_t", "phi_x", "c_x"]],
                dtype=jnp.float64
            )
        else:
            res_all = jnp.abs(Residual.get_res_PHYS(model, x_norm, t_norm))
            N = res_all.shape[0] // 2
            criterion_lsts = jnp.asarray([res_all[:N], res_all[N:]], dtype=jnp.float64)

        idx_list = []
        for lst in criterion_lsts:
            top_idx = jnp.argpartition(-jnp.abs(lst), n_adapt)[:n_adapt]
            idx_list.append(top_idx)

        merged = jnp.concatenate(idx_list)
        unique_indices = jnp.unique(merged, size=n_adapt, fill_value=merged[0])
        return merged

    @staticmethod
    def compute_jacobian(res_func, model, x, t):
        params = eqx.filter(model, eqx.is_array)
        flat_params, reconstruct = ravel_pytree(params)

        def flat_residuals(flat_params):
            new_params = reconstruct(flat_params)
            new_model = eqx.combine(new_params, model)
            return res_func(new_model, x, t).ravel()

        return jax.jacfwd(flat_residuals)(flat_params)

    @staticmethod
    def compute_ntk_weights(model, x, t):
        eps = 1e-8
        x_phys, t_phys = x.getval('colloc', 'adapt'), t.getval('colloc', 'adapt')
        x_ic, t_ic = x.getval('ic'), t.getval('ic')
        x_bc, t_bc = x.getval('bc'), t.getval('bc')

        J_phys = Residual.compute_jacobian(Residual.get_res_PHYS, model, x_phys, t_phys)
        N_phys = x_phys.shape[0]

        J_ic = Residual.compute_jacobian(Residual.get_res_IC, model, x_ic, t_ic)
        J_bc = Residual.compute_jacobian(Residual.get_res_BC, model, x_bc, t_bc)

        tr_ac = jnp.sum(J_phys[:N_phys]**2) + eps
        tr_ch = jnp.sum(J_phys[N_phys:]**2) + eps
        tr_ic = jnp.sum(J_ic**2) + eps
        tr_bc = jnp.sum(J_bc**2) + eps

        n_ac = jnp.float64(N_phys)
        n_ch = jnp.float64(N_phys)
        n_ic = jnp.float64(J_ic.shape[0])
        n_bc = jnp.float64(J_bc.shape[0])

        S = (tr_ac / n_ac) + (tr_ch / n_ch) + (tr_ic / n_ic) + (tr_bc / n_bc)
        weights = S * jnp.asarray([n_ac / tr_ac, n_ch / tr_ch, n_ic / tr_ic, n_bc / tr_bc], dtype=jnp.float64)
        return weights

# ----------------------debug------------------------
    # def compute_ntk_weights(self, model, x, t):
    #     eps = 1e-8  # safe denominator
    
    #     # --- Unpack ---
    #     x_phys, t_phys = x.getval('colloc', 'adapt'), t.getval('colloc', 'adapt')
    #     x_ic, t_ic = x.getval('ic'), t.getval('ic')
    #     x_bc, t_bc = x.getval('bc'), t.getval('bc')
    
    #     # --- Debug: show coordinates ---
    #     jax.debug.print("\n[NTK DEBUG] x_phys = {}", x_phys)
    #     jax.debug.print("[NTK DEBUG] t_phys = {}", t_phys)
    #     jax.debug.print("[NTK DEBUG] x_ic = {}", x_ic)
    #     jax.debug.print("[NTK DEBUG] t_ic = {}", t_ic)
    #     jax.debug.print("[NTK DEBUG] x_bc = {}", x_bc)
    #     jax.debug.print("[NTK DEBUG] t_bc = {}", t_bc)
    
    #     # --- Compute Jacobians ---
    #     J_phys = self.compute_jacobian(self.get_res_PHYS, model, x_phys, t_phys)
    #     N_phys = x_phys.shape[0]
    
    #     J_ic = self.compute_jacobian(self.get_res_IC, model, x_ic, t_ic)
    #     J_bc = self.compute_jacobian(self.get_res_BC, model, x_bc, t_bc)
    
    #     # --- Traces ---
    #     tr_ac = jnp.sum(J_phys[:N_phys] ** 2)
    #     tr_ch = jnp.sum(J_phys[N_phys:] ** 2)
    #     tr_ic = jnp.sum(J_ic ** 2)
    #     tr_bc = jnp.sum(J_bc ** 2)
    
    #     # --- Counts ---
    #     n_ac = N_phys
    #     n_ch = N_phys
    #     n_ic = 2 * x_ic.shape[0]
    #     n_bc = 2 * x_bc.shape[0]
    
    #     # --- Compute S and weights ---
    #     S = (
    #         (tr_ac / (n_ac + eps)) +
    #         (tr_ch / (n_ch + eps)) +
    #         (tr_ic / (n_ic + eps)) +
    #         (tr_bc / (n_bc + eps))
    #     )
    
    #     weights = S * jnp.asarray([
    #         n_ac / (tr_ac + eps),
    #         n_ch / (tr_ch + eps),
    #         n_ic / (tr_ic + eps),
    #         n_bc / (tr_bc + eps)
    #     ])
    
    #     # --- Debug: show traces, counts, S, weights ---
    #     jax.debug.print(
    #         "[NTK DEBUG] tr_ac={:.3e}, tr_ch={:.3e}, tr_ic={:.3e}, tr_bc={:.3e}",
    #         tr_ac, tr_ch, tr_ic, tr_bc
    #     )
    #     jax.debug.print(
    #         "[NTK DEBUG] n_ac={}, n_ch={}, n_ic={}, n_bc={}",
    #         n_ac, n_ch, n_ic, n_bc
    #     )
    #     jax.debug.print(
    #         "[NTK DEBUG] S={:.3e}, weights={}",
    #         S, weights
    #     )
    
    #     return weights

    # --debug--    
    # @staticmethod
    # def get_res_PHYS(model, x, t):
    #     derivs = Residual._denormalized_derivatives(model, x, t)
    
    #     h_val = h(derivs["phi"])
    #     h_phi = h_p(derivs["phi"])
    #     h_xx = h_pp(derivs["phi"]) * derivs["phi_x"]**2 + h_p(derivs["phi"]) * derivs["phi_xx"]
    #     g_phi = g_p(derivs["phi"])
    
    #     # Debug prints for all components of res_phi
    #     debug.print("phi_t: {x}", x=derivs["phi_t"])
    #     debug.print("c: {x}", x=derivs["c"])
    #     debug.print("h_val: {x}", x=h_val)
    #     debug.print("dc: {x}", x=dc)
    #     debug.print("c_l: {x}", x=c_l)
    #     debug.print("h_phi: {x}", x=h_phi)
    #     debug.print("g_phi: {x}", x=g_phi)
    #     debug.print("phi_xx: {x}", x=derivs["phi_xx"])
    #     x1 = -2.0 * A * L * (derivs["c"] - h_val * dc - c_l) * dc * h_phi
    #     x2 = L * omega_phi * g_phi
    #     x3 = -L * alpha_phi * derivs["phi_xx"]
    #     debug.print("phi_t: {x}", x=derivs["phi_t"])
    #     debug.print("x1:{}",x1)
    #     debug.print("x2:{}",x2)
    #     debug.print("x3:{}",x3)
        
    #     res_phi = (
    #         derivs["phi_t"]
    #         - 2.0 * A * L * (derivs["c"] - h_val * dc - c_l) * dc * h_phi
    #         + L * omega_phi * g_phi
    #         - L * alpha_phi * derivs["phi_xx"]
    #     )
    
    #     res_c = (
    #         derivs["c_t"]
    #         - 2.0 * A * M * derivs["c_xx"]
    #         + 2.0 * A * M * dc * h_xx
    #     )
    
    #     return jnp.concatenate([
    #         jnp.asarray(res_phi, dtype=jnp.float64),
    #         jnp.asarray(res_c, dtype=jnp.float64)
    #     ])