include("XRD_Module.jl")
using .XRD_Module: main

main("./data/XRD_data.txt", "./output/XRD_results.csv")