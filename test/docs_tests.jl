import Documenter
import ModelPNPS

# The manual is executed by `docs/make.jl`; this group keeps the fast edit loop focused on
# docstrings and avoids evaluating manual `CurrentModule` blocks outside the docs process.
Documenter.doctest(ModelPNPS; manual = false)
