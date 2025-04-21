import jax.numpy as jnp

import unxt as u
import coordinax as cx
import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

w = gc.PhaseSpaceCoordinate(
    q=u.Quantity([8, 0, 0], "kpc"),
    p=u.Quantity([0, 220, 0], "km/s"),
    t=u.Quantity(0, "Myr"),
)

pot = gp.MilkyWayPotential2022()

orbit = gd.evaluate_orbit(pot, w, u.Quantity(jnp.linspace(0, 35, 35), "Myr"))

print(orbit[-1])
