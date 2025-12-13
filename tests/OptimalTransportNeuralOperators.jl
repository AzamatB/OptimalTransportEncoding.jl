using GeometryBasics
using FileIO
using MeshIO

M = Matrix{Float32}
file_path = "datasets/ShapeNet-Car/data/mesh_001.ply"
mesh = load(file_path)
measure = OrientedSurfaceMeasure{M}(mesh)

n = 64
torus = Torus(n)
measure_latent = LatentOrientedSurfaceMeasure{M}(torus)

@time (encoding_indices, decoding_indices, ot_plan, measure_transported) = compute_encoder_and_decoder(
    measure, measure_latent
)

unique(encoding_indices)
unique(decoding_indices)

encoded_points = measure.points[:, encoding_indices]
decoded_points = measure_transported.points[:, decoding_indices]

count(iszero, ot_plan.plan)/length(ot_plan.plan)
minimum(sum(ot_plan.plan; dims=2)) / (1 / size(ot_plan.plan, 1))
minimum(sum(ot_plan.plan; dims=1)) / (1 / size(ot_plan.plan, 2))
