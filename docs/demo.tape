# Re-render from docs/demo: create a venv there, pip install "widemem-ai[local]", make sure Ollama is running with llama3.2:3b pulled, then run: vhs ../demo.tape

Output ../demo.gif
Set Shell "zsh"
Set FontSize 18
Set Width 1000
Set Height 480
Set Theme "Catppuccin Mocha"
Set PlaybackSpeed 2.5

Hide
Type "source .venv/bin/activate" Enter
Type "setopt interactivecomments" Enter
Type "export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore" Enter
Type "clear" Enter
Show

Type "python remember.py" Enter
Sleep 27s
Type "clear" Enter
Type "# new process, nothing in RAM" Enter
Sleep 1s
Type "python recall.py" Enter
Sleep 11s
