import cogsworth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax.numpy as jnp
import astropy.units as uas
import unxt as u
import coordinax as cx
import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

p = cogsworth.pop.Population(100, final_kstar1=[13, 14])
p.BSE_settings.update({"kickflag": 1})
p.sample_initial_binaries()
p.sample_initial_galaxy()
p.perform_stellar_evolution()

v_phi = p.initial_galaxy.v_T / p.initial_galaxy.rho
v_X = (
    p.initial_galaxy.v_R * np.cos(p.initial_galaxy.phi)
    - p.initial_galaxy.rho * np.sin(p.initial_galaxy.phi) * v_phi
)
v_Y = (
    p.initial_galaxy.v_R * np.sin(p.initial_galaxy.phi)
    + p.initial_galaxy.rho * np.cos(p.initial_galaxy.phi) * v_phi
)

# combine the representation and differentials into a Gala PhaseSpacePosition
w0s = gc.PhaseSpaceCoordinate(
    q=jnp.array(
        [
            a.to(uas.kpc).value
            for a in [p.initial_galaxy.x, p.initial_galaxy.y, p.initial_galaxy.z]
        ]
    ).T
    * u.unit("kpc"),
    p=jnp.array(
        [a.to(uas.km / uas.s).value for a in [v_X, v_Y, p.initial_galaxy.v_z]]
    ).T
    * u.unit("km / s"),
    t=jnp.zeros(len(p.initial_galaxy.x)) * u.unit("Myr"),
)
pot = gp.MilkyWayPotential2022()
pre_SN_orbit = gd.evaluate_orbit(pot, w0s, u.Quantity(jnp.linspace(0, 100, 100), "Myr"))
usys = u.unitsystem("kpc", "Msun", "Myr", "rad")

w0s_before_kick = pre_SN_orbit[-1]
kick = cx.CartesianVel3D(
    x=jnp.array(np.random.uniform(50, 150, (len(v_Y),))) * u.unit("km / s"),
    y=jnp.array(np.random.uniform(50, 150, (len(v_Y),))) * u.unit("km / s"),
    z=jnp.array(np.random.uniform(50, 150, (len(v_Y),))) * u.unit("km / s"),
)
w0s_after_kick = gc.PhaseSpaceCoordinate(
    q=w0s_before_kick.q,
    p=w0s_before_kick.p + kick.uconvert(usys),
    t=w0s_before_kick.t,
)

print("kicked orbit calculation")
post_SN_orbit = gd.evaluate_orbit(
    pot, w0s_after_kick, u.Quantity(jnp.linspace(100, 500, 400), "Myr")
)
post_SN_orbit = post_SN_orbit[-1]
print("unkicked orbit calculation")
unkicked_orbit = gd.evaluate_orbit(
    pot, w0s_before_kick, u.Quantity(jnp.linspace(100, 500, 400), "Myr")
)
unkicked_orbit = unkicked_orbit[-1]
# Plotting the orbits
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# print(post_SN_orbit.q.x.value)
kwargs = dict(density=True, bins=30, histtype="step", lw=2)
ax[0].hist(post_SN_orbit.q.x.value, **kwargs)
ax[0].hist(unkicked_orbit.q.x.value, **kwargs)
ax[0].set_xlabel("X (kpc)")
ax[0].set_ylabel("Counts")
ax[1].hist(post_SN_orbit.q.y.value, **kwargs)
ax[1].hist(unkicked_orbit.q.y.value, **kwargs)
ax[1].set_xlabel("Y (kpc)")
ax[2].hist(post_SN_orbit.q.z.value, label="kicked", **kwargs)
ax[2].hist(unkicked_orbit.q.z.value, label="Un-kicked", **kwargs)
ax[2].set_xlabel("Z (kpc)")
ax[2].legend()
fig.tight_layout()
fig.savefig("orbit.jpg")

# Plotting the orbits
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.hist(
    jnp.sqrt(
        post_SN_orbit.q.x.value ** 2 + post_SN_orbit.q.y.value ** 2 + post_SN_orbit.q.z.value ** 2
        ),
        **kwargs)
ax.set_xlabel("R (kpc)")
ax.hist(
    jnp.sqrt(
        unkicked_orbit.q.x.value ** 2
        + unkicked_orbit.q.y.value ** 2
        + unkicked_orbit.q.z.value ** 2
    ),
    **kwargs,
)
ax.set_ylabel("Counts")
fig.tight_layout()
fig.savefig("orbit_r.jpg")