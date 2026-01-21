defmodule ExMaxsimCpu.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/mixedbread-ai/maxsim-cpu"

  def project do
    [
      app: :ex_maxsim_cpu,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),

      # Docs
      name: "ExMaxsimCpu",
      source_url: @source_url,
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:rustler, "~> 0.37.0"},
      {:nx, "~> 0.7"},
      {:torchx, "~> 0.7", optional: true},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:benchee, "~> 1.0", only: :dev}
    ]
  end

  defp description do
    """
    High-performance MaxSim (Maximum Similarity) scoring using BLAS GEMM operations.
    Elixir bindings for the maxsim-cpu Rust library with SIMD acceleration.
    """
  end

  defp package do
    [
      name: "ex_maxsim_cpu",
      files: ~w(lib native .formatter.exs mix.exs README.md LICENSE),
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "ExMaxsimCpu",
      extras: ["README.md"]
    ]
  end
end
