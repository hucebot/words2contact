# Constraining the Output of Local LLMs to a Valid JSON File

Using the Python bindings of [llama.cpp](https://github.com/ggerganov/llama.cpp), it is possible to constrain the output of an LLM to follow a well-defined grammar. This is achieved by providing a grammar file to the `llama.cpp` library. The grammar file is written in the [GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) format.

If you are not comfortable writing your own GBNF file, you can use the [Grammar Builder Tool](https://grammar.intrinsiclabs.ai/) by Intrinsic Labs to generate `llama.cpp`-compatible grammar files from simple TypeScript interfaces.