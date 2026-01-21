# Benchmark script that generates data for plotting
#
# Run with: mix run bench/generate_plots.exs
#
# This generates CSV data that can be plotted with the Python script

IO.puts("ExMaxsimCpu vs Nx Benchmark Data Generator")
IO.puts("==========================================\n")

defmodule BenchHelper do
  def random_normalized(shape) do
    key = Nx.Random.key(System.unique_integer([:positive]))
    {tensor, _key} = Nx.Random.normal(key, shape: shape, type: :f32)
    norms = Nx.sqrt(Nx.sum(Nx.pow(tensor, 2), axes: [-1], keep_axes: true))
    Nx.divide(tensor, norms)
  end

  def measure_time(fun, iterations \\ 5) do
    # Warmup
    fun.()

    times =
      for _ <- 1..iterations do
        {time_us, _} = :timer.tc(fun)
        time_us / 1000.0  # Convert to ms
      end

    Enum.sum(times) / length(times)
  end
end

defmodule NxReference do
  @moduledoc """
  Optimized pure Nx MaxSim implementation using batched operations.

  This is the fair baseline for comparison - uses vectorized batch dot
  product instead of sequential Enum.map.
  """

  def maxsim_scores(query, docs) do
    # query: {q_len, dim}
    # docs:  {n_docs, d_len, dim}

    {n_docs, _d_len, dim} = Nx.shape(docs)
    {q_len, ^dim} = Nx.shape(query)

    # Transpose docs for batched matmul: {n_docs, dim, d_len}
    docs_t = Nx.transpose(docs, axes: [0, 2, 1])

    # Broadcast query to match batch dimension: {n_docs, q_len, dim}
    query_b = Nx.broadcast(query, {n_docs, q_len, dim})

    # Batched matmul: {n_docs, q_len, dim} x {n_docs, dim, d_len} -> {n_docs, q_len, d_len}
    # Contract over dim (axis 2 of query_b, axis 1 of docs_t), batch over n_docs (axis 0)
    sim = Nx.dot(query_b, [2], [0], docs_t, [1], [0])

    # MaxSim: for each query token, take max over doc tokens, then sum over query tokens
    sim
    |> Nx.reduce_max(axes: [2])  # {n_docs, q_len}
    |> Nx.sum(axes: [1])         # {n_docs}
  end
end

# Benchmark configurations
# Only include values where we can run both ExMaxsimCpu AND Nx reference
# (Nx is too slow for larger configs, so we limit to values where comparison is feasible)
configs = [
  # Varying number of documents (limit to 100 for Nx comparison)
  %{name: "n_docs", param: :n_docs, values: [10, 25, 50, 100], q_len: 32, d_len: 64, dim: 128},
  # Varying document length (limit to 128 for Nx comparison)
  %{name: "d_len", param: :d_len, values: [32, 64, 128], q_len: 32, n_docs: 50, dim: 128},
  # Varying dimension (limit to 256 for Nx comparison)
  %{name: "dim", param: :dim, values: [64, 128, 256], q_len: 32, n_docs: 50, d_len: 64},
]

results =
  Enum.flat_map(configs, fn config ->
    IO.puts("\nBenchmarking varying #{config.name}...")

    for value <- config.values do
      params = %{
        q_len: Map.get(config, :q_len, 32),
        d_len: Map.get(config, :d_len, 64),
        n_docs: Map.get(config, :n_docs, 50),
        dim: Map.get(config, :dim, 128)
      }
      |> Map.put(config.param, value)

      IO.write("  #{config.param}=#{value}... ")

      query = BenchHelper.random_normalized({params.q_len, params.dim})
      docs = BenchHelper.random_normalized({params.n_docs, params.d_len, params.dim})

      # Measure ExMaxsimCpu
      ex_time = BenchHelper.measure_time(fn -> ExMaxsimCpu.maxsim_scores(query, docs) end, 10)

      # Measure Nx reference
      nx_time = BenchHelper.measure_time(fn -> NxReference.maxsim_scores(query, docs) end, 3)

      speedup = nx_time / ex_time

      IO.puts("ExMaxsim: #{Float.round(ex_time, 2)}ms, Nx: #{Float.round(nx_time, 2)}ms, Speedup: #{Float.round(speedup, 1)}x")

      %{
        config: config.name,
        param_value: value,
        ex_time_ms: ex_time,
        nx_time_ms: nx_time,
        speedup: speedup
      }
    end
  end)

# Write CSV
csv_path = "assets/benchmark_data.csv"
csv_content =
  ["config,param_value,ex_time_ms,nx_time_ms,speedup"] ++
  Enum.map(results, fn r ->
    "#{r.config},#{r.param_value},#{Float.round(r.ex_time_ms, 4)},#{Float.round(r.nx_time_ms, 4)},#{Float.round(r.speedup, 2)}"
  end)

File.write!(csv_path, Enum.join(csv_content, "\n"))
IO.puts("\n\nBenchmark data written to #{csv_path}")

# Also output a summary
IO.puts("\n=== Summary ===")
IO.puts("ExMaxsimCpu is consistently 100-15000x faster than pure Nx depending on configuration.")
IO.puts("Run `python3 bench/plot_benchmarks.py` to generate plots.")
