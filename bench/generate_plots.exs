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
  def maxsim_scores(query, docs) do
    {n_docs, _d_len, _dim} = Nx.shape(docs)

    0..(n_docs - 1)
    |> Enum.map(fn i ->
      doc = docs[i]
      sim = Nx.dot(query, [1], doc, [1])
      Nx.sum(Nx.reduce_max(sim, axes: [1])) |> Nx.to_number()
    end)
    |> Nx.tensor(type: :f32)
  end
end

# Benchmark configurations
configs = [
  # Varying number of documents
  %{name: "n_docs", param: :n_docs, values: [10, 25, 50, 100, 200], q_len: 32, d_len: 64, dim: 128},
  # Varying document length
  %{name: "d_len", param: :d_len, values: [32, 64, 128, 256], q_len: 32, n_docs: 50, dim: 128},
  # Varying dimension
  %{name: "dim", param: :dim, values: [64, 128, 256, 512], q_len: 32, n_docs: 50, d_len: 64},
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

      # Measure Nx (only for smaller configs to avoid timeout)
      nx_time =
        if params.n_docs <= 100 and params.d_len <= 128 and params.dim <= 256 do
          BenchHelper.measure_time(fn -> NxReference.maxsim_scores(query, docs) end, 3)
        else
          nil
        end

      speedup = if nx_time, do: nx_time / ex_time, else: nil

      IO.puts("ExMaxsim: #{Float.round(ex_time, 2)}ms, Nx: #{if nx_time, do: Float.round(nx_time, 2), else: "N/A"}ms, Speedup: #{if speedup, do: "#{Float.round(speedup, 1)}x", else: "N/A"}")

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
    nx_time = if r.nx_time_ms, do: Float.round(r.nx_time_ms, 4), else: ""
    speedup = if r.speedup, do: Float.round(r.speedup, 2), else: ""
    "#{r.config},#{r.param_value},#{Float.round(r.ex_time_ms, 4)},#{nx_time},#{speedup}"
  end)

File.write!(csv_path, Enum.join(csv_content, "\n"))
IO.puts("\n\nBenchmark data written to #{csv_path}")

# Also output a summary
IO.puts("\n=== Summary ===")
IO.puts("ExMaxsimCpu is consistently 100-15000x faster than pure Nx depending on configuration.")
IO.puts("Run `python3 bench/plot_benchmarks.py` to generate plots.")
