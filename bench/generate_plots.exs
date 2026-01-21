# Benchmark script that generates data for plotting
#
# Run with: mix run bench/generate_plots.exs
#
# This generates CSV data that can be plotted with the Python script
#
# For MPS benchmarks, ensure torchx is installed:
#   mix deps.get

IO.puts("ExMaxsimCpu vs Nx Benchmark Data Generator")
IO.puts("==========================================\n")

defmodule BenchHelper do
  # Suppress compile warnings for optional Torchx
  @compile {:no_warn_undefined, Torchx}
  @compile {:no_warn_undefined, EXLA}

  def random_normalized(shape, backend \\ Nx.BinaryBackend) do
    key = Nx.Random.key(System.unique_integer([:positive]))
    {tensor, _key} = Nx.Random.normal(key, shape: shape, type: :f32)
    norms = Nx.sqrt(Nx.sum(Nx.pow(tensor, 2), axes: [-1], keep_axes: true))
    result = Nx.divide(tensor, norms)

    # Transfer to specified backend if different
    if backend != Nx.BinaryBackend do
      Nx.backend_transfer(result, backend)
    else
      result
    end
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

  def torchx_mps_backend do
    with true <- Code.ensure_loaded?(Torchx),
         {:ok, _apps} <- Application.ensure_all_started(:torchx),
         true <- mps_available?() do
      {Torchx.Backend, device: :mps}
    else
      _ -> nil
    end
  end

  def torchx_cpu_backend do
    with true <- Code.ensure_loaded?(Torchx),
         {:ok, _apps} <- Application.ensure_all_started(:torchx) do
      {Torchx.Backend, device: :cpu}
    else
      _ -> nil
    end
  end

  def exla_cpu_backend do
    with true <- Code.ensure_loaded?(EXLA),
         {:ok, _apps} <- Application.ensure_all_started(:exla) do
      {EXLA.Backend, client: :host}
    else
      _ -> nil
    end
  end

  def cpu_backend do
    case exla_cpu_backend() do
      nil ->
        case torchx_cpu_backend() do
          nil -> nil
          backend -> {"Torchx (CPU)", backend}
        end

      backend ->
        {"EXLA (CPU)", backend}
    end
  end

  defp mps_available? do
    # Use Torchx's built-in device availability check
    try do
      Code.ensure_loaded?(Torchx) and Torchx.device_available?(:mps)
    rescue
      _ -> false
    end
  end

  def sync_gpu do
    # Force GPU synchronization by materializing output
    :ok
  end
end

defmodule NxReference do
  @moduledoc """
  Optimized pure Nx MaxSim implementation using batched operations.
  Works with any Nx backend (BinaryBackend, Torchx, etc.)
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
    sim = Nx.dot(query_b, [2], [0], docs_t, [1], [0])

    # MaxSim: for each query token, take max over doc tokens, then sum over query tokens
    sim
    |> Nx.reduce_max(axes: [2])  # {n_docs, q_len}
    |> Nx.sum(axes: [1])         # {n_docs}
  end
end

# Check for MPS availability
mps_backend = BenchHelper.torchx_mps_backend()
cpu_backend = BenchHelper.cpu_backend()

if mps_backend do
  IO.puts("✓ Torchx MPS backend available - will benchmark GPU acceleration")
else
  IO.puts("⚠ Torchx MPS not available - skipping GPU benchmarks")
  IO.puts("  (Install torchx and run on Apple Silicon for MPS benchmarks)")
end

if cpu_backend do
  {cpu_label, _backend} = cpu_backend
  IO.puts("✓ #{cpu_label} backend available - will benchmark Nx CPU backend")
else
  IO.puts("⚠ EXLA/Torchx CPU backend not available - skipping Nx CPU benchmarks")
  IO.puts("  (Install exla or torchx to benchmark Nx CPU backends)")
end

IO.puts("")

# One-time warmup to avoid NIF load cost in the first measurement
IO.puts("Warming up NIF...")
warmup_query = BenchHelper.random_normalized({2, 2})
warmup_docs = BenchHelper.random_normalized({1, 2, 2})
_ = ExMaxsimCpu.maxsim_scores(warmup_query, warmup_docs)

if cpu_backend do
  {_label, backend} = cpu_backend
  warmup_query_cpu = Nx.backend_transfer(warmup_query, backend)
  warmup_docs_cpu = Nx.backend_transfer(warmup_docs, backend)
  _ = NxReference.maxsim_scores(warmup_query_cpu, warmup_docs_cpu) |> Nx.to_binary()
end

# Benchmark configurations
# Only include values where we can run comparisons without timeout
configs = [
  # Varying number of documents
  %{name: "n_docs", param: :n_docs, values: [10, 25, 50, 100], q_len: 32, d_len: 64, dim: 128},
  # Varying document length
  %{name: "d_len", param: :d_len, values: [32, 64, 128], q_len: 32, n_docs: 50, dim: 128},
  # Varying dimension
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

      # Generate data on BinaryBackend first
      query_bin = BenchHelper.random_normalized({params.q_len, params.dim})
      docs_bin = BenchHelper.random_normalized({params.n_docs, params.d_len, params.dim})

      # Measure ExMaxsimCpu (uses BinaryBackend tensors, converts internally)
      ex_time = BenchHelper.measure_time(fn ->
        ExMaxsimCpu.maxsim_scores(query_bin, docs_bin)
      end, 10)

      # Measure Nx BinaryBackend
      nx_time = BenchHelper.measure_time(fn ->
        result = NxReference.maxsim_scores(query_bin, docs_bin)
        _ = Nx.to_binary(result)  # Force materialization
        :ok
      end, 3)

      # Measure Nx CPU backend (EXLA/Torchx) if available
      {nx_cpu_time, nx_cpu_backend_label} =
        if cpu_backend do
          {label, backend} = cpu_backend
          query_cpu = Nx.backend_transfer(query_bin, backend)
          docs_cpu = Nx.backend_transfer(docs_bin, backend)

          time = BenchHelper.measure_time(fn ->
            result = NxReference.maxsim_scores(query_cpu, docs_cpu)
            _ = Nx.to_binary(result)
            :ok
          end, 3)

          {time, label}
        else
          {nil, nil}
        end

      # Measure Nx with MPS backend (if available)
      {mps_time, mps_transfer_time} =
        if mps_backend do
          # Pre-transfer data to GPU
          query_mps = Nx.backend_transfer(query_bin, mps_backend)
          docs_mps = Nx.backend_transfer(docs_bin, mps_backend)

          # Warmup GPU
          _ = NxReference.maxsim_scores(query_mps, docs_mps) |> Nx.to_binary()

          # Compute-only time (data already on GPU)
          compute_time = BenchHelper.measure_time(fn ->
            result = NxReference.maxsim_scores(query_mps, docs_mps)
            _ = Nx.to_binary(result)  # Force sync
            :ok
          end, 5)

          # End-to-end time (including transfer)
          e2e_time = BenchHelper.measure_time(fn ->
            q = Nx.backend_transfer(query_bin, mps_backend)
            d = Nx.backend_transfer(docs_bin, mps_backend)
            result = NxReference.maxsim_scores(q, d)
            _ = Nx.to_binary(result)
            :ok
          end, 3)

          {compute_time, e2e_time}
        else
          {nil, nil}
        end

      # Calculate speedups
      speedup_vs_nx = nx_time / ex_time
      speedup_vs_mps = if mps_time, do: mps_time / ex_time, else: nil
      speedup_vs_nx_cpu = if nx_cpu_time, do: nx_cpu_time / ex_time, else: nil

      # Print results
      mps_str = if mps_time, do: ", MPS: #{Float.round(mps_time, 2)}ms", else: ""
      cpu_str =
        if nx_cpu_time do
          ", Nx CPU (#{nx_cpu_backend_label}): #{Float.round(nx_cpu_time, 2)}ms"
        else
          ""
        end

      IO.puts(
        "ExMaxsim: #{Float.round(ex_time, 2)}ms, Nx: #{Float.round(nx_time, 2)}ms#{cpu_str}#{mps_str}"
      )

      %{
        config: config.name,
        param_value: value,
        ex_time_ms: ex_time,
        nx_time_ms: nx_time,
        nx_cpu_time_ms: nx_cpu_time,
        nx_cpu_backend: nx_cpu_backend_label,
        mps_time_ms: mps_time,
        mps_transfer_time_ms: mps_transfer_time,
        speedup_vs_nx: speedup_vs_nx,
        speedup_vs_nx_cpu: speedup_vs_nx_cpu,
        speedup_vs_mps: speedup_vs_mps
      }
    end
  end)

# Write CSV
csv_path = "assets/benchmark_data.csv"
csv_header =
  "config,param_value,ex_time_ms,nx_time_ms,nx_cpu_time_ms,nx_cpu_backend,mps_time_ms,mps_transfer_time_ms,speedup_vs_nx,speedup_vs_nx_cpu,speedup_vs_mps"

csv_content =
  [csv_header] ++
  Enum.map(results, fn r ->
    mps_time = if r.mps_time_ms, do: Float.round(r.mps_time_ms, 4), else: ""
    mps_transfer = if r.mps_transfer_time_ms, do: Float.round(r.mps_transfer_time_ms, 4), else: ""
    nx_cpu_time = if r.nx_cpu_time_ms, do: Float.round(r.nx_cpu_time_ms, 4), else: ""
    nx_cpu_backend = r.nx_cpu_backend || ""
    speedup_nx_cpu = if r.speedup_vs_nx_cpu, do: Float.round(r.speedup_vs_nx_cpu, 2), else: ""
    speedup_mps = if r.speedup_vs_mps, do: Float.round(r.speedup_vs_mps, 2), else: ""

    "#{r.config},#{r.param_value},#{Float.round(r.ex_time_ms, 4)},#{Float.round(r.nx_time_ms, 4)},#{nx_cpu_time},#{nx_cpu_backend},#{mps_time},#{mps_transfer},#{Float.round(r.speedup_vs_nx, 2)},#{speedup_nx_cpu},#{speedup_mps}"
  end)

File.write!(csv_path, Enum.join(csv_content, "\n"))
IO.puts("\n\nBenchmark data written to #{csv_path}")

# Summary
IO.puts("\n=== Summary ===")
avg_speedup_nx = Enum.sum(Enum.map(results, & &1.speedup_vs_nx)) / length(results)
IO.puts("Average speedup vs Nx (BinaryBackend): #{Float.round(avg_speedup_nx, 0)}x")

if cpu_backend do
  {cpu_label, _backend} = cpu_backend
  cpu_results = Enum.filter(results, & &1.speedup_vs_nx_cpu)
  if length(cpu_results) > 0 do
    avg_speedup_nx_cpu =
      Enum.sum(Enum.map(cpu_results, & &1.speedup_vs_nx_cpu)) / length(cpu_results)
    IO.puts("Average speedup vs Nx CPU (#{cpu_label}): #{Float.round(avg_speedup_nx_cpu, 1)}x")
  end
end

if mps_backend do
  mps_results = Enum.filter(results, & &1.speedup_vs_mps)
  if length(mps_results) > 0 do
    avg_speedup_mps = Enum.sum(Enum.map(mps_results, & &1.speedup_vs_mps)) / length(mps_results)
    IO.puts("Average speedup vs Nx (MPS GPU): #{Float.round(avg_speedup_mps, 1)}x")
  end
end

IO.puts("\nRun `uv run bench/plot_benchmarks.py` to generate plots.")
