defmodule HelloDiffusionTest do
  use ExUnit.Case
  doctest HelloDiffusion

  test "greets the world" do
    assert HelloDiffusion.hello() == :world
  end
end
