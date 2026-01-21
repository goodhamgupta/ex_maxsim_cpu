# Benchmarks for ExMaxsimCpu
#
# Run with: mix run bench/benchmark.exs
#
# Make sure to set environment variables for optimal performance:
#   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=8 mix run bench/benchmark.exs

IO.puts("ExMaxsimCpu Benchmarks")
IO.puts("======================\n")

# Configuration
dims = [128, 256, 768]
q_lens = [32, 64]
d_lens = [128, 256, 512]
n_docs_list = [100, 1000]

# Generate random normalized vectors
defmodule BenchHelper do
  def random_normalized(shape) do
    tensor = Nx.random_normal(shape, type: :f32)
    norms = Nx.sqrt(Nx.sum(Nx.pow(tensor, 2), axes: [-1], keep_axes: true))
    Nx.divide(tensor, norms)
  end

  def format_throughput(docs_per_sec) do
    cond do
      docs_per_sec >= 1_000_000 -> "#{Float.round(docs_per_sec / 1_000_000, 2)}M docs/s"
      docs_per_sec >= 1_000 -> "#{Float.round(docs_per_sec / 1_000, 2)}K docs/s"
      true -> "#{Float.round(docs_per_sec, 2)} docs/s"
    end
  end
end

# Run benchmarks
benchmarks =
  for dim <- dims, q_len <- q_lens, d_len <- d_lens, n_docs <- n_docs_list do
    name = "dim=#{dim}, q=#{q_len}, d=#{d_len}, n=#{n_docs}"

    query = BenchHelper.random_normalized({q_len, dim})
    docs = BenchHelper.random_normalized({n_docs, d_len, dim})

    {name,
     fn ->
       ExMaxsimCpu.maxsim_scores(query, docs)
     end}
  end
  |> Enum.into(%{})

IO.puts("Running benchmarks with #{map_size(benchmarks)} configurations...\n")

Benchee.run(
  benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0.5,
  formatters: [
    {Benchee.Formatters.Console, extended_statistics: true}
  ]
)

# Also run a quick comparison with pure Nx reference implementation
IO.puts("\n\nComparison with Pure Nx Reference")
IO.puts("==================================\n")

defmodule NxReference do
  @doc """
  Pure Nx MaxSim implementation for correctness verification.
  """
  def maxsim_scores(query, docs) do
    # query: {q_len, dim}
    # docs: {n_docs, d_len, dim}
    {n_docs, d_len, dim} = Nx.shape(docs)
    {q_len, ^dim} = Nx.shape(query)

    # Reshape for batch matmul
    # query: {1, q_len, dim} -> broadcast to {n_docs, q_len, dim}
    query_expanded = Nx.reshape(query, {1, q_len, dim})

    # For each doc, compute similarity matrix and reduce
    Nx.map(docs, [axes: [0]], fn doc ->
      # doc: {d_len, dim}
      # sim = query @ doc.T -> {q_len, d_len}
      sim = Nx.dot(query, [1], doc, [1])
      # max per query, then sum
      Nx.sum(Nx.reduce_max(sim, axes: [1]))
    end)
  end
end

# Small test to verify correctness
query = BenchHelper.random_normalized({32, 128})
docs = BenchHelper.random_normalized({10, 64, 128})

our_scores = ExMaxsimCpu.maxsim_scores(query, docs)
nx_scores = NxReference.maxsim_scores(query, docs)

diff = Nx.abs(Nx.subtract(our_scores, nx_scores))
max_diff = Nx.reduce_max(diff) |> Nx.to_number()

IO.puts("Correctness check:")
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
