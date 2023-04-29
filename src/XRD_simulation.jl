include("XRD_Module.jl")

using .XRD_Module: main

@time main("./data/XRD_data.txt", "./output/XRD_results.csv")