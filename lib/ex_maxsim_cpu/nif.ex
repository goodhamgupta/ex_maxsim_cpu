defmodule ExMaxsimCpu.Nif do
  @moduledoc false
  # Internal NIF wrapper module. Do not use directly.

  use Rustler,
    otp_app: :ex_maxsim_cpu,
    crate: :maxsim_cpu

  @doc false
  @spec maxsim_scores_nif(
          binary(),
          pos_integer(),
          pos_integer(),
          binary(),
          pos_integer(),
          pos_integer()
        ) ::
          binary()
  def maxsim_scores_nif(_query_bin, _q_len, _dim, _docs_bin, _n_docs, _d_len) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  @spec maxsim_scores_variable_nif(binary(), pos_integer(), pos_integer(), [binary()], [
          pos_integer()
        ]) ::
          binary()
  def maxsim_scores_variable_nif(_query_bin, _q_len, _dim, _doc_bins, _doc_lens) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
