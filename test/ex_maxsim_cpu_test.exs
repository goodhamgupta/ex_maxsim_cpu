defmodule ExMaxsimCpuTest do
  use ExUnit.Case, async: true

  describe "maxsim_scores/2" do
    test "computes correct scores for identity vectors" do
      # Query: 2 tokens, dim 2 (identity-like)
      query = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)

      # 1 document: 2 tokens, dim 2 (same as query)
      docs = Nx.tensor([[[1.0, 0.0], [0.0, 1.0]]], type: :f32)

      scores = ExMaxsimCpu.maxsim_scores(query, docs)

      assert Nx.shape(scores) == {1}
      # Each query token's max sim is 1.0, sum = 2.0
      assert_all_close(scores, Nx.tensor([2.0], type: :f32))
    end

    test "computes scores for multiple documents" do
      query = Nx.tensor([[1.0, 0.0]], type: :f32)

      docs =
        Nx.tensor(
          [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.5, 0.5]]
          ],
          type: :f32
        )

      scores = ExMaxsimCpu.maxsim_scores(query, docs)

      assert Nx.shape(scores) == {3}
      # doc 0: max(1.0*1.0 + 0.0*0.0) = 1.0
      # doc 1: max(1.0*0.0 + 0.0*1.0) = 0.0
      # doc 2: max(1.0*0.5 + 0.0*0.5) = 0.5
      assert_all_close(scores, Nx.tensor([1.0, 0.0, 0.5], type: :f32))
    end

    test "handles larger dimensions" do
      dim = 128
      q_len = 32
      d_len = 64
      n_docs = 10

      query = Nx.broadcast(1.0 / :math.sqrt(dim), {q_len, dim}) |> Nx.as_type(:f32)
      docs = Nx.broadcast(1.0 / :math.sqrt(dim), {n_docs, d_len, dim}) |> Nx.as_type(:f32)

      scores = ExMaxsimCpu.maxsim_scores(query, docs)

      assert Nx.shape(scores) == {n_docs}
      # All docs should have similar scores
      scores_list = Nx.to_flat_list(scores)
      assert Enum.all?(scores_list, &(&1 > 0))
    end

    test "raises on dimension mismatch" do
      query = Nx.tensor([[1.0, 0.0, 0.0]], type: :f32)
      docs = Nx.tensor([[[1.0, 0.0]]], type: :f32)

      assert_raise ArgumentError, ~r/Dimension mismatch/, fn ->
        ExMaxsimCpu.maxsim_scores(query, docs)
      end
    end

    test "raises on invalid query shape" do
      query = Nx.tensor([1.0, 0.0], type: :f32)
      docs = Nx.tensor([[[1.0, 0.0]]], type: :f32)

      assert_raise ArgumentError, ~r/Invalid query shape/, fn ->
        ExMaxsimCpu.maxsim_scores(query, docs)
      end
    end
  end

  describe "maxsim_scores_variable/2" do
    test "computes correct scores for variable length docs" do
      query = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)

      doc1 = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
      doc2 = Nx.tensor([[0.5, 0.5]], type: :f32)

      scores = ExMaxsimCpu.maxsim_scores_variable(query, [doc1, doc2])

      assert Nx.shape(scores) == {2}
      # doc1: both query tokens find perfect match -> 2.0
      # doc2: single token matches both queries at 0.5 -> 1.0
      assert_all_close(scores, Nx.tensor([2.0, 1.0], type: :f32))
    end

    test "handles empty list" do
      query = Nx.tensor([[1.0, 0.0]], type: :f32)

      assert_raise ArgumentError, ~r/Empty document list not supported/, fn ->
        ExMaxsimCpu.maxsim_scores_variable(query, [])
      end
    end
  end

  # Helper to compare tensors with tolerance
  defp assert_all_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    rtol = Keyword.get(opts, :rtol, 1.0e-4)

    diff = Nx.subtract(actual, expected) |> Nx.abs()
    tolerance = Nx.add(atol, Nx.multiply(rtol, Nx.abs(expected)))

    assert Nx.all(Nx.less_equal(diff, tolerance)) |> Nx.to_number() == 1,
           "Tensors not close:\nActual: #{inspect(Nx.to_flat_list(actual))}\nExpected: #{inspect(Nx.to_flat_list(expected))}"
  end
end
