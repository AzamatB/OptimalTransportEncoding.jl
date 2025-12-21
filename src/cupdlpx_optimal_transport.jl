using cuPDLPx.LibcuPDLPx

function form_optimal_transport_lp(
    costs::CuMatrix{Float32}, μ::Vector{Float64}, ν::Vector{Float64}
)
    (n, m) = Int32.(size(costs))
    @assert length(μ) == n
    @assert length(ν) == m
    num_vars = n * m
    num_cons = n + m

    # objective coefficients
    c = Vector{Float64}(undef, num_vars)
    copyto!(c, vec(costs))
    # constant term in objective
    c₀ = Float64[0.0]

    # equality constraints: con_lb == con_ub == [μ; ν]
    rhs = [μ; ν]
    con_lb = rhs
    con_ub = copy(rhs)

    # variable bounds: x >= 0
    var_lb = zeros(Float64, num_vars)
    var_ub = fill(Inf64, num_vars)

    # CSR structure: num_cons x num_vars, nnz = 2 * num_vars
    nnz = Int32(2 * num_vars)
    row_ptr = Vector{Int32}(undef, num_cons + 1)
    row_ptr[1] = one(Int32)
    @inbounds for i in 1:n
        row_ptr[i+1] = row_ptr[i] + m
    end
    @inbounds for j in 1:m
        row_ptr[n+j+1] = row_ptr[n+j] + n
    end

    col_ind = Vector{Int32}(undef, nnz)
    one_to_num_vars = Int32(1):Int32(num_vars)
    # source constraints block (rows 1..n): row-wise access in column-major flattening
    @. col_ind[one_to_num_vars] = linear_index(one_to_num_vars, n, m)
    # target constraints block (rows n+1..n+m): contiguous columns
    # col_ind[(num_vars+1):(2 * num_vars)] = 1:num_vars
    col_ind[(num_vars+1):end] .= one_to_num_vars

    # values are all ones
    vals = ones(Float64, nnz)
    coefficients = (; c, con_lb, con_ub, var_lb, var_ub, c₀, nnz, row_ptr, col_ind, vals)
    return coefficients
end

function linear_index(k::Int32, n::Int32, m::Int32)
    uno = one(Int32)
    km1 = k - uno
    # im1 = i-1   (0-based row index in row-major enumeration)
    im1 = km1 ÷ m
    # jm1 = j-1   (0-based col index)
    jm1 = km1 - im1 * m
    # 1-based column-major linear index: i + (j-1)*n
    col_index = im1 + uno + jm1 * n
    return col_index
end

function extract_plan(x::CuVector{Float32}, n::Int, m::Int)
    return reshape(x, n, m)
end
