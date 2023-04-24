using Random
using DataFrames
using CSV

include("XRD_Module.jl")

using .XRD_Module: do_it_zero, do_it

function main(data_file_name)
    Random.seed!(347) # Setting the seed for random noise

    θ₀ = do_it_zero(data_file_name)
    df = DataFrame(θ=θ₀, SC=θ₀, BCC=θ₀, FCC=θ₀)
    
    for lattice_type in ("SC", "BCC", "FCC")
        df[:, "θ"], df[:, lattice_type] = do_it(data_file_name, lattice_type)
    end
    
    CSV.write("./output/XRD_results.csv", df)
end


main("simple_XRD.txt")