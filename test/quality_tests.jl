using Test: @testset

import Aqua
import ExplicitImports
import JET
import ModelPNPS

@testset "test_package_quality" begin
    Aqua.test_all(ModelPNPS)
    ExplicitImports.check_no_implicit_imports(ModelPNPS)
    ExplicitImports.check_no_stale_explicit_imports(ModelPNPS)
    ExplicitImports.check_all_qualified_accesses_via_owners(ModelPNPS)
    # Package-wide analysis with abstract dependency entry points produces dozens of
    # upstream HDF5/Luna/Base reports. Keep every report whose origin is ModelPNPS; the
    # concrete integration tests exercise the dependency calls themselves.
    JET.test_package(ModelPNPS; target_modules = (ModelPNPS,))
end
