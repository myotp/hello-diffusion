defmodule HelloDiffusion.MixProject do
  use Mix.Project

  def project do
    [
      app: :hello_diffusion,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {HelloDiffusion.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      {:exla, github: "elixir-nx/nx", sparse: "exla", override: true},
      {:image, "~> 0.42.0"},
      {:kino, "~> 0.12.3"},
      {:bumblebee, github: "elixir-nx/bumblebee", override: true}
    ]
  end
end
