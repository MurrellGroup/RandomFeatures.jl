using RandomFeatures
using Test

@testset "RandomFeatures.jl" begin
    rff = RandomFourierFeatures(10=>20,0.1f0)
    x = randn(Float32,10,11,1)
    @test isapprox(rff(x[:,:]) , rff(x)[:,:,1])

    rof = RandomOrientationFeatures(16,0.1f0)
    @test size(rof((randn(3,3,10,1),randn(3,1,10,1)))) == (16, 10, 10, 1)
end
