defmodule ExMaxsimCpu do
  @moduledoc """
  High-performance MaxSim (Maximum Similarity) scoring using BLAS GEMM operations.

  MaxSim computes the maximum similarity score between a query and a set of documents.
  For each query token, it finds the maximum similarity with any document token,
  then sums these maximums to produce the final score.

  ## Usage with Nx Tensors

      # Query: [q_len, dim] tensor
      query = Nx.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], type: :f32)

      # Documents: [n_docs, d_len, dim] tensor
      docs = Nx.tensor([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9], [0.1, 0.0, 0.1]]
      ], type: :f32)

      scores = ExMaxsimCpu.maxsim_scores(query, docs)
      # => #Nx.Tensor<f32[2]>

  ## Usage with Raw Binaries

  For advanced use cases, you can use raw binaries directly:

      query_bin = <<0.1::float-32-native, 0.2::float-32-native, ...>>
      docs_bin = <<...>>

      scores = ExMaxsimCpu.maxsim_scores_raw(query_bin, q_len, dim, docs_bin, n_docs, d_len)

  ## Performance Notes

  - Uses Dirty CPU schedulers for compute-intensive operations
  - Leverages BLAS (OpenBLAS on Linux, Accelerate on macOS) for matrix operations
  - SIMD-optimized max reduction (AVX2 on x86_64, NEON on ARM64)
  - Rayon for parallel document processing

  ## Environment Variables

  - `RAYON_NUM_THREADS`: Control Rayon parallelism (default: number of CPUs)
  - `OPENBLAS_NUM_THREADS`: Set to 1 to avoid oversubscription with Rayon
  """

  alias ExMaxsimCpu.Nif

  @doc """
  Compute MaxSim scores between a query and a batch of documents.

  ## Parameters

  - `query`: An Nx tensor of shape `{q_len, dim}` with type `:f32`
  - `docs`: An Nx tensor of shape `{n_docs, d_len, dim}` with type `:f32`

  ## Returns

  An Nx tensor of shape `{n_docs}` containing the MaxSim score for each document.

  ## Examples

      iex> query = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
      iex> docs = Nx.tensor([[[1.0, 0.0], [0.0, 1.0]]], type: :f32)
      iex> ExMaxsimCpu.maxsim_scores(query, docs)
      #Nx.Tensor<
        f32[1]
        [2.0]
      >
  """
  @spec maxsim_scores(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def maxsim_scores(query, docs) do
    # Validate shapes
    {q_len, q_dim} = validate_query_shape(query)
    {n_docs, d_len, d_dim} = validate_docs_shape(docs)

    if q_dim != d_dim do
      raise ArgumentError,
            "Dimension mismatch: query dim #{q_dim} vs docs dim #{d_dim}"
    end

    # Ensure f32 type and contiguous layout
    query = Nx.as_type(query, :f32)
    docs = Nx.as_type(docs, :f32)

    query_bin = Nx.to_binary(query)
    docs_bin = Nx.to_binary(docs)

    scores_bin = Nif.maxsim_scores_nif(query_bin, q_len, q_dim, docs_bin, n_docs, d_len)

    Nx.from_binary(scores_bin, :f32)
  end

  @doc """
  Compute MaxSim scores for variable-length documents.

  ## Parameters

  - `query`: An Nx tensor of shape `{q_len, dim}` with type `:f32`
  - `docs`: A list of Nx tensors, each with shape `{doc_len_i, dim}` with type `:f32`

  ## Returns

  An Nx tensor of shape `{length(docs)}` containing the MaxSim score for each document.
  """
  @spec maxsim_scores_variable(Nx.Tensor.t(), [Nx.Tensor.t()]) :: Nx.Tensor.t()
  def maxsim_scores_variable(_query, []) do
    raise ArgumentError, "Empty document list not supported (Nx cannot create empty tensors). Use maxsim_scores_variable_raw/5 for empty lists."
  end

  def maxsim_scores_variable(query, docs) when is_list(docs) do
    {q_len, q_dim} = validate_query_shape(query)

    # Validate and convert all documents
    {doc_bins, doc_lens} =
      docs
      |> Enum.with_index()
      |> Enum.map(fn {doc, idx} ->
        shape = Nx.shape(doc)

        case shape do
          {doc_len, dim} when dim == q_dim ->
            doc = Nx.as_type(doc, :f32)
            {Nx.to_binary(doc), doc_len}

          {_, dim} ->
            raise ArgumentError,
                  "Dimension mismatch at doc #{idx}: query dim #{q_dim} vs doc dim #{dim}"

          other ->
            raise ArgumentError,
                  "Invalid doc shape at #{idx}: expected {doc_len, dim}, got #{inspect(other)}"
        end
      end)
      |> Enum.unzip()

    query = Nx.as_type(query, :f32)
    query_bin = Nx.to_binary(query)

    scores_bin = Nif.maxsim_scores_variable_nif(query_bin, q_len, q_dim, doc_bins, doc_lens)

    Nx.from_binary(scores_bin, :f32)
  end

  @doc """
  Compute MaxSim scores using raw binaries (advanced API).

  This is a lower-level API for users who want to avoid Nx tensor overhead.

  ## Parameters

  - `query_bin`: Binary containing query vectors as f32 values (native endian)
  - `q_len`: Number of query tokens
  - `dim`: Embedding dimension
  - `docs_bin`: Binary containing document vectors as f32 values
  - `n_docs`: Number of documents
  - `d_len`: Number of tokens per document (must be uniform)

  ## Returns

  Binary containing n_docs f32 scores.
  """
  @spec maxsim_scores_raw(binary(), pos_integer(), pos_integer(), binary(), pos_integer(), pos_integer()) ::
          binary()
  def maxsim_scores_raw(query_bin, q_len, dim, docs_bin, n_docs, d_len)
      when is_binary(query_bin) and is_binary(docs_bin) and
             is_integer(q_len) and q_len > 0 and
             is_integer(dim) and dim > 0 and
             is_integer(n_docs) and n_docs > 0 and
             is_integer(d_len) and d_len > 0 do
    expected_query_size = q_len * dim * 4
    expected_docs_size = n_docs * d_len * dim * 4

    if byte_size(query_bin) != expected_query_size do
      raise ArgumentError,
            "Query binary size mismatch: expected #{expected_query_size}, got #{byte_size(query_bin)}"
    end

    if byte_size(docs_bin) != expected_docs_size do
      raise ArgumentError,
            "Docs binary size mismatch: expected #{expected_docs_size}, got #{byte_size(docs_bin)}"
    end

    Nif.maxsim_scores_nif(query_bin, q_len, dim, docs_bin, n_docs, d_len)
  end

  @doc """
  Compute MaxSim scores for variable-length documents using raw binaries (advanced API).

  ## Parameters

  - `query_bin`: Binary containing query vectors as f32 values
  - `q_len`: Number of query tokens
  - `dim`: Embedding dimension
  - `doc_bins`: List of binaries, each containing a document's vectors
  - `doc_lens`: List of token counts for each document

  ## Returns

  Binary containing n_docs f32 scores.
  """
  @spec maxsim_scores_variable_raw(binary(), pos_integer(), pos_integer(), [binary()], [pos_integer()]) ::
          binary()
  def maxsim_scores_variable_raw(_query_bin, _q_len, _dim, [], []), do: <<>>

  def maxsim_scores_variable_raw(query_bin, q_len, dim, doc_bins, doc_lens)
      when is_binary(query_bin) and is_list(doc_bins) and is_list(doc_lens) do
    expected_query_size = q_len * dim * 4

    if byte_size(query_bin) != expected_query_size do
      raise ArgumentError,
            "Query binary size mismatch: expected #{expected_query_size}, got #{byte_size(query_bin)}"
    end

    if length(doc_bins) != length(doc_lens) do
      raise ArgumentError, "doc_bins and doc_lens must have the same length"
    end

    # Validate each document binary size
    Enum.zip(doc_bins, doc_lens)
    |> Enum.with_index()
    |> Enum.each(fn {{bin, len}, idx} ->
      expected = len * dim * 4

      if byte_size(bin) != expected do
        raise ArgumentError,
              "Doc #{idx} binary size mismatch: expected #{expected}, got #{byte_size(bin)}"
      end
    end)

    Nif.maxsim_scores_variable_nif(query_bin, q_len, dim, doc_bins, doc_lens)
  end

  # Private helpers

  defp validate_query_shape(query) do
    case Nx.shape(query) do
      {q_len, dim} ->
        {q_len, dim}

      other ->
        raise ArgumentError,
              "Invalid query shape: expected {q_len, dim}, got #{inspect(other)}"
    end
  end

  defp validate_docs_shape(docs) do
    case Nx.shape(docs) do
      {n_docs, d_len, dim} ->
        {n_docs, d_len, dim}

      other ->
        raise ArgumentError,
              "Invalid docs shape: expected {n_docs, d_len, dim}, got #{inspect(other)}"
    end
  end
end
