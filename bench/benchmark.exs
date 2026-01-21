# Benchmarks for ExMaxsimCpu
#
# Run with: mix run bench/benchmark.exs
#
# Make sure to set environment variables for optimal performance:
#   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=8 mix run bench/benchmark.exs

IO.puts("ExMaxsimCpu Benchmarks")
IO.puts("======================\n")

# Generate random normalized vectors
defmodule BenchHelper do
  def random_normalized(shape) do
    key = Nx.Random.key(System.unique_integer([:positive]))
    {tensor, _key} = Nx.Random.normal(key, shape: shape, type: :f32)
    norms = Nx.sqrt(Nx.sum(Nx.pow(tensor, 2), axes: [-1], keep_axes: true))
    Nx.divide(tensor, norms)
  end
end

defmodule NxReference do
  @moduledoc """
  Pure Nx MaxSim implementation for correctness verification.
  """
  def maxsim_scores(query, docs) do
    # query: {q_len, dim}
    # docs: {n_docs, d_len, dim}
    {n_docs, _d_len, _dim} = Nx.shape(docs)

    # Process each document
    0..(n_docs - 1)
    |> Enum.map(fn i ->
      doc = docs[i]
      # sim = query @ doc.T -> {q_len, d_len}
      sim = Nx.dot(query, [1], doc, [1])
      # max per query, then sum
      Nx.sum(Nx.reduce_max(sim, axes: [1])) |> Nx.to_number()
    end)
    |> Nx.tensor(type: :f32)
  end
end

# Test data
IO.puts("Generating test data...")
query = BenchHelper.random_normalized({32, 128})
docs = BenchHelper.random_normalized({100, 64, 128})

# Correctness check
IO.puts("\nCorrectness check:")
our_scores = ExMaxsimCpu.maxsim_scores(query, docs)
nx_scores = NxReference.maxsim_scores(query, docs)

diff = Nx.abs(Nx.subtract(our_scores, nx_scores))
max_diff = Nx.reduce_max(diff) |> Nx.to_number()

IO.puts("  Max difference from Nx reference: #{Float.round(max_diff, 8)}")
IO.puts("  Status: #{if max_diff < 1.0e-4, do: "✓ PASS", else: "✗ FAIL"}")

# Benchmark comparison
IO.puts("\nPerformance comparison (32 query tokens, 128 dim, 64 doc tokens, 100 docs):")

Benchee.run(
  %{
    "ExMaxsimCpu (BLAS+SIMD)" => fn ->
      ExMaxsimCpu.maxsim_scores(query, docs)
    end,
    "Pure Nx Reference" => fn ->
      NxReference.maxsim_scores(query, docs)
    end
  },
  warmup: 1,
  time: 3
)

# Larger scale benchmark (ExMaxsimCpu only - Nx reference is too slow)
IO.puts("\n\nLarger scale benchmark (ExMaxsimCpu only):")
IO.puts("Configuration: 32 query tokens, 768 dim, 128 doc tokens, 1000 docs")
query_large = BenchHelper.random_normalized({32, 768})
docs_large = BenchHelper.random_normalized({1000, 128, 768})

Benchee.run(
  %{
    "ExMaxsimCpu (BLAS+SIMD)" => fn ->
      ExMaxsimCpu.maxsim_scores(query_large, docs_large)
    end
  },
  warmup: 0.5,
  time: 2
)
