# H200 pod agent guide — moved

The brief for the on-pod agent now covers **both** GPU suites in one document, because
they run on the same card in the same session and their results only make sense together
(the modal transform is launch-bound, this one saturates the card — the contrast is the
point).

**Read instead:** `/workspace/code/Luna.jl/test/manual/H200_AGENT_BRIEF.md`
(in this repo's sibling checkout; `Luna.jl/test/manual/H200_AGENT_BRIEF.md`).

Suite B in that brief is this repo's `examples/h200_modelpnps_suite.sh`, driven by
`examples/h200_bench.jl` and `examples/h200_scan_rehearsal.jl`.
